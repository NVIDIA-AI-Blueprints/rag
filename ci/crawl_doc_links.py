#!/usr/bin/env python3
"""
Crawl a documentation site starting from an index URL.
Follow internal links up to a given depth; report 4xx/5xx and redirects.
Usage: python ci/crawl_doc_links.py <index_url> [--depth N]
"""

import argparse
import re
import sys
from urllib.parse import urljoin, urlparse

import requests

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; DocLinkCrawler/1.0)"})


def get_internal_links(html: str, base_url: str) -> set[str]:
    """Extract links from HTML that are internal (same host as base_url)."""
    base_host = urlparse(base_url).netloc
    links = set()
    for m in re.finditer(r'href\s*=\s*["\']([^"\']+)["\']', html, re.I):
        raw = m.group(1).strip()
        if raw.startswith(("#", "javascript:", "mailto:")):
            continue
        full = urljoin(base_url, raw)
        parsed = urlparse(full)
        if parsed.netloc == base_host or not parsed.netloc:
            no_frag = full.split("#")[0] or full
            links.add(no_frag)
    return links


def fetch(url: str, timeout: int):
    """GET url with redirects. Return (final_url, status_code, history, body, error)."""
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True, verify=False)
        return (r.url, r.status_code, r.history, r.text, None)
    except requests.exceptions.RequestException as e:
        return (url, None, [], "", str(e))

def main():
    parser = argparse.ArgumentParser(description="Crawl doc site and report broken/redirect links")
    parser.add_argument("index_url", help="Index URL (e.g. https://.../rag/latest/index.html)")
    parser.add_argument("--depth", type=int, default=2, help="Max depth (default 2)")
    parser.add_argument("--timeout", type=int, default=15, help="Request timeout seconds")
    args = parser.parse_args()

    base_url = args.index_url.rstrip("/")
    base_host = urlparse(base_url).netloc
    if not base_host:
        print("Invalid index URL", file=sys.stderr)
        sys.exit(1)

    # BFS: (url, depth, found_on_url)
    to_visit = [(base_url, 0, None)]
    seen = {base_url}
    broken = []   # (url, status_or_error, found_on)
    redirects = []  # (url, status, final_url, found_on)

    while to_visit:
        url, depth, found_on = to_visit.pop(0)
        final_url, status, history, body, err = fetch(url, args.timeout)

        if err:
            broken.append((url, err, found_on))
            continue
        if status >= 400:
            broken.append((url, f"HTTP {status}", found_on))
            continue
        if history:
            redirects.append((url, history[0].status_code, final_url, found_on))

        if depth >= args.depth:
            continue
        if status != 200:
            continue

        internal = get_internal_links(body, final_url)
        for link in internal:
            if link not in seen:
                seen.add(link)
                to_visit.append((link, depth + 1, url))

    # Report
    print("=" * 60)
    print("BROKEN (4xx/5xx or fetch error)")
    print("=" * 60)
    if not broken:
        print("(none)")
    else:
        for url, reason, found_on in sorted(broken, key=lambda x: (x[0], x[2] or "")):
            where = f" (found on: {found_on})" if found_on else " (index)"
            print(f"  {reason}: {url}{where}")

    print()
    print("=" * 60)
    print("REDIRECTS")
    print("=" * 60)
    if not redirects:
        print("(none)")
    else:
        for url, status, final_url, found_on in sorted(redirects, key=lambda x: (x[0], x[3] or "")):
            where = f" (found on: {found_on})" if found_on else " (index)"
            print(f"  {status} {url} -> {final_url}{where}")

    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
