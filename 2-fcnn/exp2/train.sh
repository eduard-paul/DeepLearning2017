#!/bin/bash

set -e

WORK_DIR=/home/epaul/ASL/
EXP_DIR=exp2
SNAPSHOT_DIR=$WORK_DIR/$EXP_DIR/asl_exp1_2
TOOLS=/home/epaul/caffe/build/tools

if [ ! -d "$SNAPSHOT_DIR" ]; then
    mkdir -p $SNAPSHOT_DIR
fi

# Train FCN
echo "Train FCN"
$TOOLS/caffe train -gpu=0 -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -log_dir=$WORK_DIR/$EXP_DIR/ $@

# Test FCN
echo "Test FCN"
$TOOLS/caffe test -gpu=0 -model=$WORK_DIR/$EXP_DIR/asl_exp1_2.prototxt -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -weights="$SNAPSHOT_DIR/asl_exp1_2_iter_10000.caffemodel" $@
