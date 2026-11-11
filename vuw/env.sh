# Source before anything on this server: every cache and temp file stays inside this directory.
export B=${B:?set B to the data root (original: the project directory on the GPU server)}
# CUDA must number GPUs as nvidia-smi does (PCI bus), or CUDA_VISIBLE_DEVICES=<nvidia-smi index> lands on another GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PIP_CACHE_DIR=$B/.cache/pip XDG_CACHE_HOME=$B/.cache/xdg TORCH_HOME=$B/.cache/torch
export YOLO_CONFIG_DIR=$B/.cache/ultralytics CUDA_CACHE_PATH=$B/.cache/cuda MPLCONFIGDIR=$B/.cache/mpl
export HF_HOME=$B/.cache/hf TMPDIR=$B/tmp TMP=$B/tmp TEMP=$B/tmp PYTHONPYCACHEPREFIX=$B/.cache/pycache
. $B/venv/bin/activate
# the GPU with the most free memory among those at <= 50% utilisation (empty if none has >= 4 GB free)
pick_gpu () {
  nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' '$4 <= 50 && $2 - $3 >= 4000 {print $2 - $3, $1}' | sort -rn | head -1 | awk '{print $2}'
}
# DEEP's code: data roots on this server (never the home directory)
export RAP_DATA_ROOT=$B/datasets RAP_NUSCENES_ROOT=$B/datasets/nuscenes RAP_NUSCENES_TRAINVAL=$B/datasets/nuscenes
export RAP_MODEL_ROOT=$B/models RAP_FRAME=ego RAP_REPO=$B/code/risk-aware-perception
export PYTHONPATH=$B/code/CVPR/src:$B/code/risk-aware-perception/src:$B/scripts/common
