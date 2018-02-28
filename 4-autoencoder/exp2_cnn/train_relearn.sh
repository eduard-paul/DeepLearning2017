#!/bin/bash

set -e

WORK_DIR=/home/epaul/ASL
EXP_DIR=lab3/exp2_cnn
SNAPSHOT_DIR=$WORK_DIR/$EXP_DIR/asl_exp3_2_relearn
TOOLS=/home/epaul/caffe/build/tools

if [ ! -d "$SNAPSHOT_DIR" ]; then
    mkdir -p $SNAPSHOT_DIR
fi

# Train CCN
echo "Train CCN"
$TOOLS/caffe train -gpu=0 -solver=$WORK_DIR/$EXP_DIR/relearn_solver.prototxt -weights="$WORK_DIR/$EXP_DIR/asl_exp3_2_autoencoder/asl_exp3_2_autoencoder_iter_2000.caffemodel" -log_dir=$WORK_DIR/$EXP_DIR/  $@

# Test CCN regression
echo "Test CCN"
$TOOLS/caffe test -gpu=0 -model=$WORK_DIR/$EXP_DIR/asl_exp3_2_relearn.prototxt -solver=$WORK_DIR/$EXP_DIR/relearn_solver.prototxt -weights="$SNAPSHOT_DIR/asl_exp3_2_relearn_iter_2000.caffemodel" -iterations=62 $@
