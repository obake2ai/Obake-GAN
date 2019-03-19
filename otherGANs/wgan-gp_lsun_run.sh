#!/bin/sh
#PBS -q l-regular
#PBS -l select=1:mpiprocs=1:ompthreads=32
#PBS -W group_list=gj16
#PBS -l walltime=100:00:00
cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh
module load cuda9/9.1.85
export PYENV_ROOT="/lustre/gj16/j16002/.pyenv/"
export PATH=$PYENV_ROOT/bin:$PATH
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi
python train_wgan-gp_lsun.py
