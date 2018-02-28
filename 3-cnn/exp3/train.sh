#!/bin/bash

set -e

WORK_DIR=/home/epaul/ASL
EXP_DIR=lab2/exp3
SNAPSHOT_DIR=$WORK_DIR/$EXP_DIR/asl_exp2_3
TOOLS=/home/epaul/caffe/build/tools

if [ ! -d "$SNAPSHOT_DIR" ]; then
    mkdir -p $SNAPSHOT_DIR
fi

# Train FCN
echo "Train FCN"
$TOOLS/caffe train -gpu=0 -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -log_dir=$WORK_DIR/$EXP_DIR/ $@

# Test FCN regression
echo "Test FCN"
$TOOLS/caffe test -gpu=0 -model=$WORK_DIR/$EXP_DIR/asl_exp2_3.prototxt -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -weights="$SNAPSHOT_DIR/asl_exp2_3_iter_12000.caffemodel" -iterations=62 $@
