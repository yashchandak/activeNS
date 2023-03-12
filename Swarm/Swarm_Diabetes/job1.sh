#!/bin/bash
#
#SBATCH --job-name=NS_1
#SBATCH --output=./stdoutput_1/NS_1_%A_%a.out # output file
#SBATCH --error=./stdoutput_1/NS_1.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=0-11:59       # Maximum runtime in D-HH:MM
#SBATCH --cpus-per-task=1    # CPU cores per process
#SBATCH --mem-per-cpu=400    # Memory in MB per cpu allocated
#SBATCH --mail-type=END
#SBATCH --mail-user=ychandak@cs.umass.edu

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

export PATH="/mnt/nfs/work1/pthomas/ychandak/miniconda3/envs/pytorch/bin:$PATH"
export PYTHONPATH="/mnt/nfs/home/ychandak/Control-NSDP:$PYTHONPATH"
source activate pytorch

python ./run_parallel.py --inc $SLURM_ARRAY_TASK_ID --base 0 --hyper 0 --speed 1

sleep 1
exit