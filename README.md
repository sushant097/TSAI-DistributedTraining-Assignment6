# Assignment 6 - Distributed Training

<div align="center">

# Data Version Control and Experiment Tracking

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description
This repository related to data tracking and experimentation. We use dvc to save the use files like logs, models in the 
separate destination like Google Drive, or any other place whereas git adds the code to the repository. 


## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/sushant097/TSAI-Assignment3-ExperimentTracking
cd TSAI-Assignment3-ExperimentTracking

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```


```
### Push model, logs and data to google drive (using dvc)
1. Untrack logs from git : `git rm -r --cached logs`
2. Add logs to dvc: `dvc add logs`
2.1. `git add . ` and `dvc config core.autostage true` : As logs folder from being tracked by git and then let dvc take care of it
3. Add a remote: `dvc remote add -d gpu-logs s3://tsai-models/ddp-training/`
4. Push logs and other tracked files by dvc in gdrive: `dvc push -r gpu-logs`
5. Now, whenever logs is deleted then, we can directly pull logs from dvc as: `dvc pull -r gpu-logs`
