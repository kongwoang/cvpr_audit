# Source for nuPlan work: the nuPlan conda env, every cache inside the project, DEEP's nuPlan paths.
export B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
export HOME_SANDBOX=$B/.home
export PIP_CACHE_DIR=$B/.cache/pip XDG_CACHE_HOME=$B/.cache/xdg TORCH_HOME=$B/.cache/torch MPLCONFIGDIR=$B/.cache/mpl
export TMPDIR=$B/tmp TMP=$B/tmp TEMP=$B/tmp PYTHONPYCACHEPREFIX=$B/.cache/pycache_nuplan RAY_TMPDIR=$B/tmp
export PATH=$B/envs/nuplan/bin:$PATH
export RAP_REPO=$B/code/risk-aware-perception RAP_FRAME=ego RAP_MODEL_ROOT=$B/models RAP_DATA_ROOT=$B/datasets
export RAP_NUPLAN_ROOT=$B/datasets/nuplan RAP_NUPLAN_DB=$B/datasets/nuplan/nuplan-v1.1/splits/mini
export RAP_NUPLAN_MAPS=$B/datasets/nuplan/nuplan-maps-v1.0 RAP_NUPLAN_CAM_F0=$B/datasets/nuplan/sensor_blobs
export NUPLAN_DATA_ROOT=$B/datasets/nuplan NUPLAN_MAPS_ROOT=$B/datasets/nuplan/nuplan-maps-v1.0 NUPLAN_EXP_ROOT=$B/results/nuplan_exp
export PYTHONPATH=$B/code/CVPR/src:$B/code/risk-aware-perception/src
export YOLO_CONFIG_DIR=$B/.cache/ultralytics TMPDIR=$B/tmp HOME=$B/.home
