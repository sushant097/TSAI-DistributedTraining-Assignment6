

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

Dump nvidia logs:
```while true; 
do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> gpu_utillization.csv; sleep 2; 
done
```
**Maximum batch size = 512**
#### Tensorboard dev logs

```
tensorboard dev upload --logdir logs \
    --name "(optional) FSDP_Training_" \
    --description "(optional) Training on 4 T4 GPUs - FSDP Training"
```
#### Tensorboard dev Experiment logs: https://tensorboard.dev/experiment/iPPND0xlTEmMbrRsaysJHg/

#### Nvidia logs:
```
utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB]
83 %, 54 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 67 %, 15360 MiB, 1952 MiB, 12957 MiB
83 %, 64 %, 15360 MiB, 1952 MiB, 12957 MiB
83 %, 43 %, 15360 MiB, 1952 MiB, 12957 MiB

56 %, 5 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 36 %, 15360 MiB, 1952 MiB, 12957 MiB
97 %, 33 %, 15360 MiB, 1952 MiB, 12957 MiB
77 %, 18 %, 15360 MiB, 1952 MiB, 12957 MiB

13 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB

9 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB

0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB

0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 53 %, 15360 MiB, 1952 MiB, 12957 MiB
98 %, 69 %, 15360 MiB, 1952 MiB, 12957 MiB
62 %, 7 %, 15360 MiB, 1952 MiB, 12957 MiB

100 %, 73 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB

71 %, 50 %, 15360 MiB, 1952 MiB, 12957 MiB
58 %, 21 %, 15360 MiB, 1952 MiB, 12957 MiB
82 %, 60 %, 15360 MiB, 1952 MiB, 12957 MiB
79 %, 59 %, 15360 MiB, 1952 MiB, 12957 MiB

85 %, 50 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 69 %, 15360 MiB, 1952 MiB, 12957 MiB
70 %, 46 %, 15360 MiB, 1952 MiB, 12957 MiB
73 %, 29 %, 15360 MiB, 1952 MiB, 12957 MiB

100 %, 71 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 65 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 66 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 77 %, 15360 MiB, 1952 MiB, 12957 MiB

100 %, 77 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 74 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 71 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 77 %, 15360 MiB, 1952 MiB, 12957 MiB

99 %, 71 %, 15360 MiB, 1952 MiB, 12957 MiB
99 %, 74 %, 15360 MiB, 1952 MiB, 12957 MiB
99 %, 70 %, 15360 MiB, 1952 MiB, 12957 MiB
99 %, 71 %, 15360 MiB, 1952 MiB, 12957 MiB

68 %, 44 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 69 %, 15360 MiB, 1952 MiB, 12957 MiB
72 %, 52 %, 15360 MiB, 1952 MiB, 12957 MiB
70 %, 51 %, 15360 MiB, 1952 MiB, 12957 MiB

100 %, 57 %, 15360 MiB, 1952 MiB, 12957 MiB
84 %, 49 %, 15360 MiB, 1952 MiB, 12957 MiB
96 %, 61 %, 15360 MiB, 1952 MiB, 12957 MiB
59 %, 26 %, 15360 MiB, 1952 MiB, 12957 MiB

100 %, 62 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 73 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 60 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 58 %, 15360 MiB, 1952 MiB, 12957 MiB

100 %, 71 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 72 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 69 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 75 %, 15360 MiB, 1952 MiB, 12957 MiB

99 %, 72 %, 15360 MiB, 1952 MiB, 12957 MiB
99 %, 74 %, 15360 MiB, 1952 MiB, 12957 MiB
99 %, 70 %, 15360 MiB, 1952 MiB, 12957 MiB
99 %, 73 %, 15360 MiB, 1952 MiB, 12957 MiB

64 %, 38 %, 15360 MiB, 1952 MiB, 12957 MiB
92 %, 65 %, 15360 MiB, 1952 MiB, 12957 MiB
68 %, 44 %, 15360 MiB, 1952 MiB, 12957 MiB
64 %, 37 %, 15360 MiB, 1952 MiB, 12957 MiB

56 %, 19 %, 15360 MiB, 1952 MiB, 12957 MiB
92 %, 52 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 62 %, 15360 MiB, 1952 MiB, 12957 MiB
55 %, 18 %, 15360 MiB, 1952 MiB, 12957 MiB

100 %, 67 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 71 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 60 %, 15360 MiB, 1952 MiB, 12957 MiB
100 %, 69 %, 15360 MiB, 1952 MiB, 12957 MiB

0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB

0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
6 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
0 %, 0 %, 15360 MiB, 1952 MiB, 12957 MiB
```
More logs is in `gpu_utilization_fsdp.csv`


### Multi-node training

**Command:**
```

MASTER_PORT=29500 MASTER_ADDR=172.31.23.109 WORLD_SIZE=2 NODE_RANK=0 python src/train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2 experiment=example_timm

MASTER_PORT=29500 MASTER_ADDR=172.31.26.101 WORLD_SIZE=2 NODE_RANK=1 python src/train.py trainer=ddp_sim trainer.devices=4 trainer.num_nodes=2 experiment=example_timm
```


**Instance Type: g4dn.12xlarge**

#### Tensorboard dev logs

```
tensorboard dev upload --logdir logs --name " Multi-node Training" --description " Training on 4 T4 GPUs each node - Multi-node Training"
```
#### Tensorboard dev Experiment logs: https://tensorboard.dev/experiment/0KEXYxL1RCmQE8QONpkmGA/#scalars&_smoothingWeight=0.81

**Maximum batch size=512**

**Effective batch size=2x512x4**