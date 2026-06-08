
## Pre-reqs for SLURM:
```bash
module load cuda/11.8.0-fasrc01
module load Mambaforge # Loaded Mambaforge/23.11.0-fasrc01
python --version # Python 3.10.13
# Test GPU (above)
```

## Common instructions
```bash
mamba create -n zfish python=3.7 -c conda-forge
mamba activate zfish
python --version # Python 3.7.12

mamba install ffmpeg
python -m pip install torch torchvision # seems to install nvidia/cuda too
python -c "import torch; print(torch.__version__)" # 1.13.1+cu117


git clone <REPO>; cd <REPO>

cat requirements.txt | xargs -n 1 python -m pip install
python -m pip install -e .
python -m pip install seaborn

cd onpolicy/custom/forage
python MAZFish.py # tests env
bash run_lambda_single.sh # test training
```