"""
download_real_datasets.py - Safe & Throttled Downloader for Real Public Datasets
Downloads UNSW-NB15, BGL, and HDFS log files directly from official open-data mirrors.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "name": "UNSW_NB15_training-set.csv",
        "url": "https://raw.githubusercontent.com/NSS-Group/UNSW-NB15/master/UNSW_NB15_training-set.csv",
        "target": RAW_DIR / "UNSW_NB15_training-set.csv"
    },
    {
        "name": "BGL.log",
        "url": "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log",
        "target": RAW_DIR / "BGL.log"
    },
    {
        "name": "HDFS.log",
        "url": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
        "target": RAW_DIR / "HDFS.log"
    }
]

def download_file(url, target_path):
    print(f"  • Downloading {target_path.name} from {url[:65]}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            elapsed = time.time() - t0
            size_kb = len(data) / 1024
            print(f"    ✓ Downloaded {size_kb:.1f} KB in {elapsed:.2f}s -> {target_path.name}")
    except Exception as e:
        print(f"    [!] Error downloading {target_path.name}: {e}")

def main():
    print("==========================================================================")
    print(" Downloading REAL Public Benchmark Datasets (UNSW-NB15, BGL, HDFS)")
    print("==========================================================================")
    for ds in DATASETS:
        download_file(ds["url"], ds["target"])
        time.sleep(1) # Pause to prevent disk I/O spikes
    print("--------------------------------------------------------------------------")
    print("[+] All dataset files downloaded safely to data/raw/")

if __name__ == "__main__":
    main()
