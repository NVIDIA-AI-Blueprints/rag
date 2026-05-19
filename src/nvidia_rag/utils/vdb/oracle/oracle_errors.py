# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate raw oracledb / OCI errors into operator-readable next steps.

A "seamless" deployment is one where, when something does go wrong, the
operator looking at ``kubectl logs`` immediately knows **what to fix** —
not "Oracle is unavailable, check your config".

Every common ORA- / OCI- code we expect to see during a fresh install or
existing-DB BYO is mapped here to:

  * a one-line **category** ("auth", "network", "wallet", "service", …)
  * a short **next step** the operator can copy-paste

Usage::

    try:
        oracledb.connect(...)
    except oracledb.Error as e:
        diag = diagnose_oracle_error(e, dsn=dsn, user=user)
        logger.error("Oracle connect failed: %s\n  fix: %s",
                     diag.summary, diag.next_step)
        raise APIError(diag.user_message, ErrorCodeMapping.SERVICE_UNAVAILABLE) from e

The mapping is comprehensive but conservative: anything we don't
recognise is returned with the original message + a generic next step,
so we never make the error WORSE than the raw library output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OracleDiag:
    """Structured diagnosis of a connect / query failure."""

    code: str            # "ORA-01017", "DPY-4011", "" (unknown)
    category: str        # "auth", "network", "wallet", "service", "schema", "other"
    summary: str         # one-line human summary, no PII / no password
    next_step: str       # exact action the operator should take
    raw_message: str     # original error text (truncated)

    @property
    def user_message(self) -> str:
        """Combined message safe to surface in API responses + pod logs."""
        return f"{self.summary} — {self.next_step}"


_ORA_CODE_RE = re.compile(r"\b(ORA-\d{4,5}|DPY-\d{4}|DPI-\d{4})\b")


# Category, summary, next_step, keyed by the (5-char) ORA code prefix.
# Sources: Oracle Database Error Messages 26ai docs, oracledb thin-mode
# DPY codes, common ADB failure modes seen in pre-prod runs.
_TABLE: dict[str, tuple[str, str, str]] = {
    # ---- Authentication / authorization ------------------------------------
    "ORA-01017": ("auth",
                  "Bad ORACLE_USER or ORACLE_PASSWORD",
                  "kubectl get secret oracle-creds -o yaml — confirm the "
                  "decoded ORACLE_USER/ORACLE_PASSWORD match the ADB user. "
                  "If you rotated, re-run the provisioner Job."),
    "ORA-28000": ("auth",
                  "ADB account is locked",
                  "Login to ADB as ADMIN and run "
                  "`ALTER USER <user> ACCOUNT UNLOCK;`"),
    "ORA-01045": ("auth",
                  "User has no CREATE SESSION privilege",
                  "Connect as ADMIN and run "
                  "`GRANT CREATE SESSION TO <user>;`"),

    # ---- Connect string / DSN ----------------------------------------------
    "ORA-12154": ("network",
                  "Connect string not in tnsnames or wallet",
                  "Check ORACLE_CS in oracle-creds; for ADB it must match a "
                  "TNS alias from the wallet (e.g. <name>_high / _medium / "
                  "_low). Verify TNS_ADMIN points at the wallet directory."),
    "ORA-12514": ("service",
                  "ADB service is starting up or stopped",
                  "OCI Console → Autonomous Database → confirm state is "
                  "AVAILABLE. A freshly-provisioned DB takes ~5 min."),

    # ---- Network / VCN routing ---------------------------------------------
    "ORA-12541": ("network",
                  "ADB private endpoint unreachable from this pod",
                  "VCN routing problem. From an OKE worker: "
                  "`getent hosts <adb-host>` should resolve to a private IP, "
                  "and `nc -zv <host> 1522` must succeed. Check the ADB "
                  "private-endpoint NSG / security list allows TCP 1522 "
                  "from the OKE worker subnet CIDR."),
    "ORA-12170": ("network",
                  "TCP connect to ADB timed out",
                  "Same as ORA-12541 but more often a security-list issue. "
                  "Open the ADB private endpoint subnet's ingress to TCP "
                  "1522 from the OKE worker subnet."),
    "ORA-12537": ("network",
                  "Connection closed by ADB before handshake",
                  "Usually a TLS / wallet mismatch. Verify the wallet you "
                  "loaded into the pod matches the ADB you're connecting to."),

    # ---- Wallet / TLS ------------------------------------------------------
    "ORA-28759": ("wallet",
                  "Wallet password missing or wrong",
                  "Set ORACLE_WALLET_PASSWORD on the pod (oracle-creds Secret) "
                  "to the password you used when downloading the wallet."),
    "ORA-28365": ("wallet",
                  "Wallet PEM file is encrypted but no password was supplied",
                  "Same fix as ORA-28759 — set ORACLE_WALLET_PASSWORD."),
    "ORA-28860": ("wallet",
                  "TLS handshake failed",
                  "Re-download the ADB wallet from OCI Console; the one in "
                  "the pod is probably stale. ADB rotates wallets."),

    # ---- Service-side --------------------------------------------------
    "ORA-00001": ("schema",
                  "Unique-constraint violation",
                  "Usually a duplicate ingest. Re-run the upload with the "
                  "`--set-replace` flag, OR `kubectl exec deploy/rag-server "
                  "-- python -c \"...\"` to delete the existing row in "
                  "the document_info table for this collection."),
    "ORA-00942": ("schema",
                  "Table or view doesn't exist",
                  "RAG_APP schema isn't bootstrapped. Re-run the provisioner "
                  "Job or run `examples/oracle/scripts/init.sql` as RAG_APP."),
    "ORA-00955": ("schema",
                  "Object name already in use",
                  "If you're re-installing, run `helm uninstall` first, OR "
                  "set --reuse-existing on the provisioner Job."),
    "ORA-65096": ("schema",
                  "Invalid common user/role name",
                  "ADB enforces CDB-friendly identifiers; pick a username "
                  "that does NOT start with `C##` and is at most 30 chars. "
                  "Update --set oracle.createDatabase.appUser=<name> and "
                  "re-run the provisioner Job."),

    # ---- thin-mode driver-level codes (DPY-/DPI-) --------------------------
    "DPY-4011": ("network",
                 "DNS lookup failed for ADB host",
                 "DNS resolution from the pod is broken. If using ADB "
                 "private endpoint, the OKE cluster's VCN must use a DNS "
                 "resolver that knows the private DNS zone "
                 "(usually `oraclevcn.com`). Check OCI VCN Resolver settings."),
    "DPY-4027": ("network",
                 "TLS connection to ADB lost mid-handshake",
                 "Wallet/cert mismatch or stale cipher suite. Re-download "
                 "wallet from OCI Console."),
    "DPI-1047": ("config",
                 "oracledb thin-mode is missing required dependency",
                 "Pod image is missing OpenSSL 1.1.1+. Use the official "
                 "rag-server image (--set rag.image.repository=...); do "
                 "not strip libssl from a custom base image."),
}


def _extract_code(message: str) -> str:
    """Return the first ORA-/DPY-/DPI- code in the message, or empty."""
    m = _ORA_CODE_RE.search(message or "")
    return m.group(1) if m else ""


def diagnose_oracle_error(
    err: BaseException,
    dsn: str | None = None,
    user: str | None = None,
) -> OracleDiag:
    """Translate any oracledb error into an actionable :class:`OracleDiag`.

    ``dsn`` and ``user`` are appended to the summary so the operator
    sees which connection parameters were in play. Passwords and
    connection strings with embedded credentials are NOT surfaced.
    """
    raw = str(err)
    code = _extract_code(raw)
    cat, summary, next_step = _TABLE.get(
        code,
        (
            "other",
            "Unrecognised Oracle error",
            "Inspect the raw message below; if the issue persists, file "
            "an issue with the redacted ORACLE_CS and the first 5 lines "
            "of `kubectl logs`.",
        ),
    )
    if user or dsn:
        suffix_bits = []
        if user:
            suffix_bits.append(f"user={user}")
        if dsn:
            # Strip any inline auth in case someone passed user/pw@host
            redacted = re.sub(r"://[^@]+@", "://***@", str(dsn))
            suffix_bits.append(f"dsn={redacted}")
        summary = f"{summary} ({', '.join(suffix_bits)})"
    return OracleDiag(
        code=code,
        category=cat,
        summary=summary,
        next_step=next_step,
        raw_message=raw[:500],
    )


# ---------------------------------------------------------------------------
# OCI provisioner errors — same shape, different code-space.
# Customers see these from the oracle-adb-provisioner Job.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OciDiag:
    code: str
    category: str
    summary: str
    next_step: str
    raw_message: str

    @property
    def user_message(self) -> str:
        return f"{self.summary} — {self.next_step}"


_OCI_HINTS: list[tuple[re.Pattern[str], tuple[str, str, str, str]]] = [
    (re.compile(r"NotAuthorizedOrNotFound|NotAuthenticated", re.I),
     ("OCI-AUTH",
      "auth",
      "OCI identity not authorized or token expired",
      "Run `oci iam region list` from the same identity to confirm. The "
      "provisioner needs at minimum: `Allow dynamic-group <dg> to manage "
      "autonomous-database-family in <compartment>; Allow dynamic-group "
      "<dg> to use virtual-network-family in <compartment>; Allow "
      "dynamic-group <dg> to read clusters in <compartment>`")),
    (re.compile(r"InvalidParameter.*displayName|already exists", re.I),
     ("OCI-409",
      "conflict",
      "ADB display name is already in use",
      "Either delete the existing DB first OR re-run the provisioner with "
      "`--reuse-existing` (Helm: --set oracle.createDatabase.reuseExisting=true)")),
    (re.compile(r"LimitExceeded|free-tier", re.I),
     ("OCI-LIMIT",
      "quota",
      "OCI tenancy ADB / cpu / storage limit reached",
      "OCI Console → Limits, Quotas and Usage. Increase the ADB count "
      "limit OR set --reuse-existing to attach to an existing instance.")),
    (re.compile(r"InvalidParameter.*subnet|SubnetNotFound", re.I),
     ("OCI-SUBNET",
      "config",
      "ADB subnet is invalid or not in the same VCN",
      "Pass --set oracle.createDatabase.subnetId=<private-subnet-OCID> "
      "where the subnet is in the same VCN as the OKE cluster.")),
    (re.compile(r"compartment.*not.*found|InvalidCompartmentId", re.I),
     ("OCI-COMPARTMENT",
      "config",
      "Compartment OCID is invalid or not visible to this identity",
      "Run `oci iam compartment list --all` and copy the exact OCID.")),
    (re.compile(r"region.*not.*subscribed|RegionNotEnabled", re.I),
     ("OCI-REGION",
      "config",
      "Region is not subscribed in this tenancy",
      "OCI Console → Regions → Subscribe. Some ADB shapes are also region-"
      "restricted; check Autonomous DB availability for your region.")),
]


def diagnose_oci_error(err: BaseException) -> OciDiag:
    raw = str(err)
    for pat, (code, cat, summary, step) in _OCI_HINTS:
        if pat.search(raw):
            return OciDiag(code, cat, summary, step, raw[:500])
    return OciDiag(
        "OCI-UNKNOWN", "other",
        "Unrecognised OCI error",
        "Check the provisioner Job logs above for the full OCI response. "
        "Most failures here are policy-related; verify the "
        "Allow statements in your dynamic group.",
        raw[:500],
    )
