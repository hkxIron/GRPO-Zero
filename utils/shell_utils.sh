#!/usr/bin/env bash

show_run_time(){
  local start_time=$1
  local end_time=$2
  local elapsed_time=$((end_time - start_time))
  # 转换为分钟和秒数
  local minutes=$((elapsed_time / 60))
  local seconds=$((elapsed_time % 60))
  echo "All job done!脚本已经运行了 ${minutes} 分钟 ${seconds} 秒。"
  echo `date`
}

test_max_gpu_num(){
  local max_gpu_num=$1 # 最少多少个gpu
  echo "限制最大gpu数量：$max_gpu_num"
  # 有时gpu里的显存有一定占用,比较小,但也可以训练
  # 或者用以下命令更好
  # nvidia-smi --query-gpu=index,memory.used --format=csv 
  # nvidia-smi --query-gpu=index,memory.used --format=csv | column -s ', ' -t
  device_list=$(nvidia-smi|grep "NVIDIA-SMI" -A37|awk -F'|' '{ print($2,"|",$3) }'|grep -E '\b[0-9]{1,3}MiB / ' -B1|grep "NVIDIA"|awk -F' ' '{print $1}'|head -n $max_gpu_num|tr '\n' ','|sed 's/,$//')
  gpu_num=$(echo ${device_list}|awk -F',' '{print NF}')
  echo "gpu ids:$device_list size:${gpu_num}"
  if [ -z "${device_list}" ];then
    echo "========================警告：没有可用gpu,退出======================="
    exit 100
  fi
}

