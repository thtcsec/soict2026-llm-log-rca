import json
from pathlib import Path
d = json.loads(Path(r"D:\tu_projects\LatexProject\soict2026\results\tables\huflit_baselines.json").read_text(encoding="utf-8"))
for name, p in d["models"].items():
    print(name)
    for k in ["precision", "recall", "f1", "roc_auc", "auprc", "latency_ms_per_seq"]:
        print(f"  {k}: {p[k]['mean']:.6f} +/- {p[k]['std']:.6f}")
