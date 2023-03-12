#!/bin/bash
#
#SBATCH --job-name=NS_3
#SBATCH --output=./stdoutput_3/NS_3_%A_%a.out # output file
#SBATCH --error=./stdoutput_3/NS_3.err        # File to which STDERR will be written
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

export PATH="/home/ychandak/miniconda3/envs/pytorch/bin:$PATH"
export PYTHONPATH="/home/ychandak/OptFuture:$PYTHONPATH"
source activate pytorch

python ./run_parallel.py --inc $SLURM_ARRAY_TASK_ID --base 3000 --hyper Diabetes3

sleep 1
exit