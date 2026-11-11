#!/usr/bin/env bash
# nuPlan simulation environment (Python 3.9, conda-forge geo stack, CPU torch) entirely inside the project directory.
# HOME is redirected so neither conda nor pip nor git writes anything to the real home directory.
set -eu
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
export HOME=$B/.home TMPDIR=$B/tmp PIP_CACHE_DIR=$B/.cache/pip CONDA_PKGS_DIRS=$B/.cache/conda_pkgs XDG_CACHE_HOME=$B/.cache/xdg
mkdir -p $HOME $TMPDIR $CONDA_PKGS_DIRS $B/envs $B/third_party
cd $TMPDIR
[ -x $B/miniforge3/bin/mamba ] || { curl -sL -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh; bash Miniforge3.sh -b -p $B/miniforge3; }
[ -x $B/envs/nuplan/bin/python ] || $B/miniforge3/bin/mamba create -y -p $B/envs/nuplan -c conda-forge --override-channels \
    python=3.9 geopandas fiona rasterio pyogrio "shapely>=2" rtree pyarrow casadi "sqlalchemy=1.4.27" "numpy=1.23" pandas scipy matplotlib
P=$B/envs/nuplan/bin/pip
$P install -q "torch>=2.0,<2.4" --index-url https://download.pytorch.org/whl/cpu
$P install -q pytorch-lightning timm "hydra-core==1.3.2" "omegaconf==2.3.1" aioboto3 aiofiles boto3 "bokeh==2.4.3" cachetools "control==0.9.1" \
    guppy3 hypothesis joblib psutil pyinstrument "pyquaternion>=0.9.5" ray requests retry sympy tornado tqdm typer ujson \
    "urllib3<1.27" opencv-python-headless pytest tensorboard testbook nest_asyncio pillow
cd $B/third_party
[ -d nuplan-devkit ] || git clone -q https://github.com/motional/nuplan-devkit.git
(cd nuplan-devkit && git checkout -q e9241677997dd86bfc0bcd44817ab04fe631405b && $P install -q -e . --no-deps)
[ -d tuplan_garage ] || git clone -q https://github.com/autonomousvision/tuplan_garage.git
(cd tuplan_garage && git checkout -q b51d5d04fac1bd4389653b9ab2ff73ea88f435a3 && $P install -q -e . --no-deps)
$B/envs/nuplan/bin/python -c "import nuplan, tuplan_garage, torch, shapely, geopandas; print('nuplan env ok', torch.__version__, shapely.__version__)"
