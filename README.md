# Mental Disorders Diagnosis from Brain Scans using Deep Learning
This is the codebase for our research on Mental Disorders. This work is a part of research course on [JeelAIDM](https://www.jeelaidm.org/)

# Environment Setup
```yaml
# clone project
git clone https://github.com/MohammedAljahdali/mental-disorder-brain-mri
cd mental-disorder-brain-mri

# [Option 1] create conda environment
conda env create -f conda_env_gpu.yaml -n myenv
conda activate myenv

# [Option 2] install the compatible pytorch and install the required packages using pip
pip install -r requirements.txt
```

# How to Train a Model
Please note that this steps assumes you already have the following [dataset](https://www.openfmri.org/dataset/ds000030/).
Then specifiy the path of the dataset in the configs/datamodule/mri_dm.yaml
## Simple Run:
`python run.py`

## Advanced Run:
`python run.py trainer=default trainer.gpus=1 trainer.precision=16 trainer.max_epochs=100 callbacks=wandb logger=wandb datamodule.batch_size=8 callbacks.log_image_predictions=None`

# Credit
This project is built on the this [template](https://github.com/ashleve/lightning-hydra-template).