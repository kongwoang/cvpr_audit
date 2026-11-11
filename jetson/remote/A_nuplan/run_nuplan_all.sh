#!/usr/bin/env bash
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
$B/scripts/A_nuplan/run_115spec.sh pdm_closed ref 6; $B/scripts/A_nuplan/run_115spec.sh pdm_closed branches 6
$B/scripts/A_nuplan/run_115spec.sh idm ref 6; $B/scripts/A_nuplan/run_115spec.sh idm branches 6
echo NUPLAN ALL DONE
