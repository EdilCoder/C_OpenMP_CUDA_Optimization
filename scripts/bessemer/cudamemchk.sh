#!/bin/bash
# Request 10 minutes of time (This should be more than enough for your assignment)
#SBATCH --time=00:10:00
# Name the job
#SBATCH --job-name=com4521_cudamemchk
# Name of file to which standard output will be redirected
#SBATCH --output="com4521_profile_output.out"
# Name of file to which the standard error stream will be redirected
#SBATCH --error="com4521_profile_error.err"
# To enable email notification, update the email address
#SBATCH --mail-user=me@somedomain.com
# Email notifications if the job begins/ends/aborts (remove the characters as desired)
# Other options are ALL or NONE
#SBATCH --mail-type=BEGIN,END,FAIL

# Request a DCS v100 test queue (max time limit 30 mins, but higher priority)
#SBATCH --partition=dcs-gpu-test
#SBATCH --account=dcs-res
#SBATCH --gres=gpu:1

# Request 4 cores
#SBATCH --cpus-per-task=4
# Request 1 gigabytes of real memory (mem) per core
#SBATCH --mem-per-cpu=1G

# Load Modules
module load CUDAcore/11.1.1
module load gcccuda/2019b

# Run the compiled program
# This example runs the Standard Deviation algorithm in CUDA mode, passing seed 12 and length 100

compute-sanitizer --print-limit 1 "./bin/release/assignment" CUDA SD 12 100