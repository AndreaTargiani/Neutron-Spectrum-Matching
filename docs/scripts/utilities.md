# Optional scripts

None of these is required to run the pipeline. Each one that already has a page also sits under **Optional** of the stage where you would actually use it; this page is the full shelf.

| Script | When you would run it |
|---|---|
| [`rebin_target_spectrum.py`](rebin_target_spectrum.md) | After `config.py`, if you want the working target on a different energy grid than the tabulated raw spectrum |
| [`bin_weights.py`](bin_weights.md) | After the working target is set, to help choose `WEIGHT_MODE` between `ICRP` and `dose` |
| [`read_opt.py`](read_opt.md) | After `optimizer.py`, to reprint a saved MOTPE study |
| [`analize_dataset.py`](analize_dataset.md) | After `dataset_creation.py`, before / alongside calibration |

[`config_materials.py`](config_materials.md) and [`config_detector.py`](config_detector.md) are part of **Configure**, not optionals.

!!! tip "Skip anything you do not need"
    The main arrow is Configure → Transfer tensors → Optimize → Validate → Dataset → Calibrate. If a stage has an **Optional** group, you can close it and keep going.
