# Raw Log Dataset Placeholder

This directory stores **raw, unprocessed log files** from HUFLIT campus infrastructure and public benchmarks.

## HUFLIT campus logs (moved, junction)

Canonical selective extract (reuse across projects):

`D:\huflit-campus-logs` ← junction target of `data/raw/huflit_logs`

Source RAR (do **not** full-extract):
`D:\huflit_logs\Thang6__CANH_BAO_giai_nen_tang_IO_disk_rat_lon.rar` (~21 GB uncompressed).
Raw ingestion is not part of the normal run. The ingestion script excludes all
FortiGate members by default and also skips members over 500 MiB.

## Expected Files (NOT committed to Git — see .gitignore)

| File | Source | Size | Format |
|---|---|---|---|
| `huflit_logs/` | HUFLIT Internal (from ThS. Cao Tiến Thành) | ~32 GB extracted + zips | FortiGate / web / SSH |
| `BGL.log.gz` | [Loghub](https://github.com/logpai/loghub) | ~744 MB | Syslog gzip |
| `HDFS_1.log` | [Loghub](https://github.com/logpai/loghub) | ~1.58 GB | HDFS plaintext |

## Public Benchmark Downloads

```bash
# BGL Supercomputer Logs
wget https://zenodo.org/record/3227177/files/BGL.tar.gz

# HDFS Distributed Cluster Logs
wget https://zenodo.org/record/3227177/files/HDFS_1.tar.gz

# UNSW-NB15 Network Traffic Logs
wget https://research.unsw.edu.au/projects/unsw-nb15-dataset

# ToN-IoT (IoT Network Logs)
wget https://research.unsw.edu.au/projects/toniot-datasets
```

## Additional Datasets (thầy gợi ý)

| Dataset | Type | Threat Scope |
|---|---|---|
| **CICIDS 2017** | Network intrusion detection | 7 attack types + Benign |
| **UNSW-NB15** | Hybrid real/synthetic network | 9 attack categories |
| **ToN_IoT** | IoT & IIoT heterogeneous logs | 9 cyber threats |
| **BoT-IoT** | IoT botnet traffic | DDoS, DoS, Recon, Theft |

Public datasets listed here are deferred and are not needed for the current paper results.
