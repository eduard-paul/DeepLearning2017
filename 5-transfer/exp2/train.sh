#!/bin/bash

set -e

WORK_DIR=/home/epaul/ASL/
EXP_DIR=lab5/exp2
SNAPSHOT_DIR=$WORK_DIR/$EXP_DIR/snapshots
TOOLS=/home/epaul/caffe/build/tools

if [ ! -d "$SNAPSHOT_DIR" ]; then
    mkdir -p $SNAPSHOT_DIR
fi

# Train FCN
echo "Train FCN"
$TOOLS/caffe train -gpu=1 -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -log_dir=$WORK_DIR/$EXP_DIR/ $@

# Test FCN
echo "Test FCN"
$TOOLS/caffe test -gpu=1 -model=$WORK_DIR/$EXP_DIR/resnet-18_asl.prototxt -solver=$WORK_DIR/$EXP_DIR/solver.prototxt -weights="$SNAPSHOT_DIR/lab5_exp2_iter_5000.caffemodel" -iterations=62 $@
