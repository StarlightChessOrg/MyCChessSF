# Local NNUE test assets

Copy a trained checkpoint here for quantize / inference smoke tests:

```powershell
copy nnue_training\checkpoints\full_gpu\best.pt tmps\best.pt
python nnue_qINT8\quantize.py --checkpoint tmps\best.pt --data-dir ..\nnue_data --data-pattern "2/worker_0/chunk_0.txt"
```

`*.pt` files in this directory are gitignored (large binaries).
