B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception
X=$B/results/navsim_exp
export OMP_NUM_THREADS=1 NAVSIM_DETS=$X/dets_final NAVSIM_PDMS_OUT=$X/pdms_final NAVSIM_TAG=_final_vs_spec_pdm NAVSIM_MAIN=spec_pdm
export NAVSIM_BRANCHES=reference,cheap,full,full960,spec,spec_pdm,uncertainty,sahi,sahi_s033,fovea
python $B/scripts/A_navsim/navsim_pdms.py --stage summary 2>&1 | grep -v PROJ
