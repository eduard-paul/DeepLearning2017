#!/bin/bash

set -e

WORK_DIR=/home/epaul/ASL
EXP_DIR=lab2/exp4
SNAPSHOT_DIR=$WORK_DIR/$EXP_DIR/asl_exp2_4
TOOLS=/home/epaul/caffe/build/tools

if [ ! -d "$SNAPSHOT_DIR" ]; then
    mkdir -p $SNAPSHOT_DIR
fi

# Train FCN
echo "Train FCN"
$TOOLS/caffe train -gpu=1 -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -log_dir=$WORK_DIR/$EXP_DIR/ $@

# Test FCN regression
echo "Test FCN"
$TOOLS/caffe test -gpu=1 -model=$WORK_DIR/$EXP_DIR/asl_exp2_4.prototxt -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -weights="$SNAPSHOT_DIR/asl_exp2_4_iter_12000.caffemodel" -iterations=62 $@
