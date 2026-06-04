# Stage 3 — NGC CLI & dataset download

Goal: the selected datasets present under `$SCRIPTS_DIR/datasets/<name>/` with the layout
`corpus/` + `train.json` (+ optional `train_extended.json` for recall).

All commands run from `$SCRIPTS_DIR` (the eval scripts dir from Stage 2) with the uv venv active.

## 3.1 Install NGC CLI (only if missing)

Check first — install only when absent:

```bash
ngc --version 2>/dev/null && echo "NGC CLI already installed" || echo "NEED_NGC_CLI"
```

If missing, install it. The authoritative, always-current command is on
<https://org.ngc.nvidia.com/setup/installers/cli> — prefer fetching the latest version string from there.
Known-good Linux AMD64 install (mirrors `run.sh`; bump the version to the latest shown on the page):

```bash
cd "$SCRIPTS_DIR"
rm -rf ngccli_linux.zip ngc-cli.md5 ngc-cli/
wget --content-disposition -q \
  "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.60.2/files/ngccli_linux.zip" \
  -O ngccli_linux.zip
unzip -q ngccli_linux.zip
chmod u+x ngc-cli/ngc
echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile && source ~/.bash_profile
ngc --version
```

(Optional integrity check: NGC publishes an `ngc-cli.md5`; verify with `md5sum -c ngc-cli.md5` if present.)

## 3.2 Configure NGC

NGC needs an API key with **`nv-rag-blueprint`** org access. Use the non-interactive path so no TTY is
required:

```bash
export NGC_CLI_API_KEY="$NGC_API_KEY"     # ngc reads NGC_CLI_API_KEY
ngc config set                            # interactive; or rely on NGC_CLI_API_KEY + --org/--team flags
```

If `ngc config set` would block on a prompt, skip it and pass `--org`/`--team` explicitly on the download
command instead, relying on `NGC_CLI_API_KEY` for auth.

## 3.3 Choose dataset versions

The eval supports many datasets (see `ALLOWED_DATASETS` in `evaluate_rag.py`). The ones in scope for this
skill and their NGC resource (org `0648981100760671`):

| Dataset (`--datasets` value) | NGC resource | ~docs / ~questions |
|------------------------------|--------------|--------------------|
| `kg_rag` | `0648981100760671/kg_rag` | 20 / 195 |
| `financebench` | `0648981100760671/financebench` | 369 / 150 |
| `hotpotqa` | `0648981100760671/hotpotqa` | 2673 / 979 |
| `google_frames` | `0648981100760671/google_frames` | 2512 / 824 |

**Always download the latest version.** List versions and pick the highest:

```bash
ngc registry resource list "0648981100760671/kg_rag" 2>/dev/null
ngc registry resource info "0648981100760671/kg_rag"     # shows available versions
```

Confirm the dataset `--datasets` name against `ALLOWED_DATASETS` before downloading (names must match
exactly — that dict also tells `evaluate_rag.py` how many documents to expect for ingestion validation).

## 3.4 Download, unzip, organize, delete the zip

For each selected dataset, download the latest version into `datasets/`. The downloaded folder is named
`<resource>_v<version>` and contains a `.zip` whose internal name can differ from the resource name:

```bash
cd "$SCRIPTS_DIR"
mkdir -p datasets

DS_NAME="kg_rag"
DS_URL="0648981100760671/kg_rag:3.0"     # use the LATEST version discovered in 3.3

ngc registry resource download-version "$DS_URL"

# download dir is "<resource>_v<version>", e.g. kg_rag_v3.0
DL_DIR=$(echo "$DS_URL" | awk -F'[:/]' '{print $2 "_v" $3}')
ZIP=$(find "$DL_DIR" -maxdepth 1 -name "*.zip" | head -1)
unzip -q "$ZIP" -d datasets
rm -f "$ZIP"                              # delete the zip as requested
rm -rf "$DL_DIR"                          # clean the NGC download wrapper dir
ls datasets/"$DS_NAME"/                   # expect: corpus  train.json  [train_extended.json]
```

Repeat per dataset. If a dataset already exists under `datasets/<name>/` with `corpus/` + `train.json`,
skip re-downloading it.

The simple form from the task prompt also works once you know the version:

```bash
ngc registry resource download-version 0648981100760671/kg_rag:3.0
unzip kg_rag.zip        # actual zip name may differ — use the find approach above when unsure
rm kg_rag.zip
```

## 3.5 Optional — recall ground truth (`train_extended.json`)

Recall@k metrics require `datasets/<name>/train_extended.json`. If a dataset ships it, keep it. If not and
recall is wanted, it can be generated with `gt_page_mapper.py` (see `scripts/README.md`) — otherwise recall
is simply skipped and the RAGAS accuracy/relevance/groundedness metrics still compute.
