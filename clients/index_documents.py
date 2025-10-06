#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Indexing-only client for POST /document
- Measures wall time
- Lets you control path/recursive/rebuild/file_types
- Prints server-reported stats (indexed_files, nodes, persist_dir, hybrid_enabled)
"""
import argparse
import json
import time
from pathlib import Path
import httpx
import sys

DEFAULT_BASE = "http://localhost:8001"

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"

def ok(msg): print(f"{GREEN}✔{RESET} {msg}")
def fail(msg): print(f"{RED}✘{RESET} {msg}")
def info(msg): print(f"{YELLOW}…{RESET} {msg}")
def bold(msg): print(f"{BOLD}{msg}{RESET}")

def main():
    ap = argparse.ArgumentParser(description="Indexing-only client for RAG server (POST /document)")
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"Server base (default: {DEFAULT_BASE})")
    ap.add_argument("--path", default='/home/datamaker/sj/project/rag/data/railway_md_0923', help="Root path (file or directory) to index")
    ap.add_argument("--recursive", action="store_true", help="Recurse directories (default: off)")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild (default: off)")
    ap.add_argument("--file-types", default="md,txt,pdf", help="Comma-separated list, e.g. md,txt")
    ap.add_argument("--timeout", type=int, default=1200, help="HTTP timeout seconds (default: 600)")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        fail(f"Path not found: {path}")
        sys.exit(1)

    payload = {
        "path": str(path),
        "recursive": bool(args.recursive),
        "rebuild": bool(args.rebuild),
        "file_types": [x.strip() for x in args.file_types.split(",") if x.strip()],
    }

    url = f"{base}/document"
    info(f"POST {url} {payload}")

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=args.timeout) as client:
            r = client.post(url, json=payload)
    except Exception as e:
        fail(f"Request error: {e}")
        sys.exit(1)
    dt = time.perf_counter() - t0

    if r.status_code != 200:
        fail(f"/document status={r.status_code} body={r.text[:400]}")
        sys.exit(1)

    try:
        data = r.json()
    except json.JSONDecodeError:
        fail("Invalid JSON response from server")
        sys.exit(1)

    if data.get("status") != "ok":
        fail(f"Server reported non-ok status: {data}")
        sys.exit(1)

    bold("\n=== Indexing Result ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    ok(f"Indexing completed in {dt:.2f}s "
       f"(indexed_files={data.get('indexed_files')}, nodes={data.get('nodes')})")

if __name__ == "__main__":
    main()
