# Evaluation:
 - change .yaml inside workspace-dir
```console
python evaluate_tracker.py --workspace_path ../workspace-dir --tracker tracker_id
```


# Vizualization:
```console
python visualize_result.py --workspace_path ../workspace-dir --tracker tracker_id --sequence basketball --show_gt
```
# Get results:
```console
python compare_trackers.py --workspace_path ../workspace-dir --trackers tracker_id --sensitivity 100
```