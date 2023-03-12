#!/usr/bin/env bash
echo "Ensure \n(a) cython_all_setup.py was executed"
echo "Ensure Experiments folder is linked to /mnt/nfs/work1/pthomas/ychandak/Experiments"
echo "Update python root path"

mkdir -p ./stdoutput_0
sbatch --array=0-15 job0.sh
#
mkdir -p ./stdoutput_1
sbatch --array=0-15 job1.sh

mkdir -p ./stdoutput_2
sbatch --array=0-15 job2.sh

# mkdir -p ./stdoutput_3
# sbatch --array=0-1000 job3.sh
#