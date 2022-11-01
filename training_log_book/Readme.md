

### Push model, logs and data to google drive (using dvc)
1. Untrack logs from git : `git rm -rf --cached training_log_book`
2. Add logs to dvc: `dvc add training_log_book`
2.1. `git add . ` and `dvc config core.autostage true` : As logs folder from being tracked by git and then let dvc take care of it
3. Add a remote: `dvc remote add -d gpu-logs s3://tsai-models/ddp-training/`
4. Push logs and other tracked files by dvc in gdrive: `dvc push -r gpu-logs`
5. Now, whenever logs is deleted then, we can directly pull logs from dvc as: `dvc pull -r gpu-logs`


# Experiments
## 1 - FSDP
`python src/train.py experiment=example_timm`

* Use `watch -n 0.5 nvidia-smi` that run `nvidia-smi` command in every 0.5 ms
