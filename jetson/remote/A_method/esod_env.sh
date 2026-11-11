# source this: the isolated ESOD environment (numpy<2, SAM), everything inside the workspace
export B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
export PIP_CACHE_DIR=$B/.cache/pip TMPDIR=$B/tmp HOME=$B/.home XDG_CACHE_HOME=$B/.cache TORCH_HOME=$B/.cache/torch \
       MPLCONFIGDIR=$B/.cache/mpl YOLO_CONFIG_DIR=$B/.cache/ultralytics
. $B/envs/esod/bin/activate
# the official YOLOv5 checkpoints are full pickles; torch >= 2.6 defaults to weights_only loading (trusted sources only)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # number GPUs as nvidia-smi does
pick_gpu () {
  nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' '$4 <= 50 && $2 - $3 >= 4000 {print $2 - $3, $1}' | sort -rn | head -1 | awk '{print $2}'
}
