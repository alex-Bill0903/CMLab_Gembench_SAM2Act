#!/bin/bash

BASE_DIR="/home/bill/Documents/research/CVPR_gembench_baseline/robot-3dlotus"

# Add the base directory itself to PYTHONPATH
export PYTHONPATH=$BASE_DIR

# Find all subdirectories up to 4 levels deep and add them to PYTHONPATH
for dir in $(find $BASE_DIR -maxdepth 4 -type d); do
    export PYTHONPATH=$PYTHONPATH:$dir
done

# Optionally, print PYTHONPATH to verify
echo $PYTHONPATH
