#!/bin/bash

# 创建日志目录
mkdir -p /mnt/hero_blob/TESTRESULTS/qwen_baselines/logs/eval_qwenimage_1105
LOGDIR=/mnt/hero_blob/TESTRESULTS/qwen_baselines/logs/eval_qwenimage_1105


nohup bash sh/eval/1105/eval_ipnomask_scut2749_qweneditplus.sh > $LOGDIR/eval_ipnomask_scut2749_qweneditplus.log 2>&1 &


nohup bash sh/eval/1105/eval_ipnomask_scutsyn_qweneditplus.sh > $LOGDIR/eval_ipnomask_scutsyn_qweneditplus.log 2>&1 &


nohup bash sh/eval/1105/eval_rectmask_ours_complex_qweneditplus.sh > $LOGDIR/eval_rectmask_ours_complex_qweneditplus.log 2>&1 &


nohup bash sh/eval/1105/eval_rectmask_ours_large_qweneditplus.sh > $LOGDIR/eval_rectmask_ours_large_qweneditplus.log 2>&1 &
