"""
inspect_logs.py - Inspects extracted HUFLIT raw campus logs
"""

import os
from pathlib import Path

def main():
    root = Path("data/raw/huflit_logs")
    if not root.exists():
        print("[!] data/raw/huflit_logs directory does not exist.")
        return

    log_files = []
    for ext in ['*.log', '*.txt', 'access', 'error', 'syslog', '*_2026*']:
        log_files.extend(root.rglob(ext))

    # Also catch all files that are not .zip or .rar
    all_files = [f for f in root.rglob('*') if f.is_file() and f.suffix.lower() not in ('.zip', '.rar', '.gitkeep')]

    total_bytes = sum(f.stat().st_size for f in all_files)
    print(f"==========================================================================")
    print(f" HUFLIT Campus Raw Log Dataset Overview")
    print(f"==========================================================================")
    print(f" Total Log Files Extracted : {len(all_files):,}")
    print(f" Total Size                : {total_bytes / 1e9:.2f} GB ({total_bytes / (1024**3):.2f} GiB)")
    print(f"--------------------------------------------------------------------------")

    # Group by service / category
    categories = {}
    for f in all_files:
        rel_path = f.relative_to(root)
        service = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        if service not in categories:
            categories[service] = {"count": 0, "bytes": 0}
        categories[service]["count"] += 1
        categories[service]["bytes"] += f.stat().st_size

    print(" Breakdown by Service / Folder:")
    for svc, info in sorted(categories.items(), key=lambda x: x[1]["bytes"], reverse=True):
        size_mb = info["bytes"] / 1e6
        print(f"  • {svc:<40} : {info['count']:>4} files | {size_mb:>8.2f} MB")
    print(f"==========================================================================")

if __name__ == "__main__":
    main()
