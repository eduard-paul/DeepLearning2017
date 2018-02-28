#!/bin/bash

set -e

WORK_DIR=/home/epaul/ASL
EXP_DIR=lab3/exp1_fcnn
SNAPSHOT_DIR=$WORK_DIR/$EXP_DIR/asl_exp3_1_autoencoder
TOOLS=/home/epaul/caffe/build/tools

if [ ! -d "$SNAPSHOT_DIR" ]; then
    mkdir -p $SNAPSHOT_DIR
fi

# Train FCN
echo "Train FCN"
$TOOLS/caffe train -gpu=1 -solver=$WORK_DIR/$EXP_DIR/autoencoder_solver.prototxt -log_dir=$WORK_DIR/$EXP_DIR/ $@

# Test FCN regression
echo "Test FCN"
$TOOLS/caffe test -gpu=0 -model=$WORK_DIR/$EXP_DIR/asl_exp3_1_autoencoder.prototxt -solver=$WORK_DIR/$EXP_DIR/autoencoder_solver.prototxt -weights="$SNAPSHOT_DIR/asl_exp3_1_autoencoder_iter_2000.caffemodel" -iterations=62 $@
