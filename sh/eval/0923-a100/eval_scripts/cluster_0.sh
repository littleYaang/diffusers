#!/bin/bash

# 创建日志目录
mkdir -p /mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/logs/eval_qwenimage_0923
LOGDIR=/mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/logs/eval_qwenimage_0923


nohup bash sh/eval/0923-a100/eval_scripts/eval_base_gs5.0.sh > $LOGDIR/eval_base_gs5.log 2>&1 &


nohup bash sh/eval/0923-a100/eval_scripts/eval_edit_gs5.0.sh > $LOGDIR/eval_edit_gs5.0.log 2>&1 &


nohup bash sh/eval/0923-a100/eval_scripts/eval_edit2509_gs5.0.sh > $LOGDIR/eval_edit2509_gs5.0.log 2>&1 &
