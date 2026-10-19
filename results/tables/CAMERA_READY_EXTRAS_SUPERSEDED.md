# Superseded artifact

`camera_ready_extras.json` compared stored OR window labels against
**template-trigger** k-of-L counts. That mixes two label constructions and must
not be cited as k-of-L aggregation sensitivity.

Use instead:

- `line_label_reconstruction_summary.json` — OR reconstruction validation
- `k_of_l_sensitivity.json` — corrected Table 6 (same raw-line proxy labels)

Commands:

```bash
python prototype/pipeline/reconstruct_line_labels.py
python prototype/pipeline/test_label_aggregation.py
python prototype/pipeline/compute_k_of_l_table.py
```
