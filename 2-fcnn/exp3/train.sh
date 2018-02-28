#!/bin/bash

set -e

WORK_DIR=/home/epaul/ASL/
EXP_DIR=exp3
SNAPSHOT_DIR=$WORK_DIR/$EXP_DIR/asl_exp1_3
TOOLS=/home/epaul/caffe/build/tools

if [ ! -d "$SNAPSHOT_DIR" ]; then
    mkdir -p $SNAPSHOT_DIR
fi

# Train FCN
echo "Train FCN"
$TOOLS/caffe train -gpu=0 -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -log_dir=$WORK_DIR/$EXP_DIR/ $@

# Test FCN regression
echo "Test FCN"
$TOOLS/caffe test -model=$WORK_DIR/$EXP_DIR/asl_exp1_3.prototxt -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -weights="$SNAPSHOT_DIR/asl_exp1_3_iter_10000.caffemodel" $@
