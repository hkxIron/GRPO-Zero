
#bash utils/scp.sh

. utils/shell_utils.sh

echo `date`
start_time=$(date +%s)
time_str="$(date +%Y%m%d-%H-%M-%S)"

root_path="$HOME/work"
project_path="${root_path}/open/project/GRPO-Zero/"
#model_path="${root_path}/open/hf_data_and_model/models/openbmb/MiniCPM3-4B/"
#model_path="${root_path}/open/hf_data_and_model/models/MoZhang96/TinyStories-LLaMA2-20M-256h-4l-GQA/"
# MoZhang96/TinyStories-LLaMA2-20M-256h-4l-GQA/

model_out_path="${root_path}/trained_models/test_grpo/"
#port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# docker image
# img1="icr"
# img2=".m"
# img3="ice.cn"
# img4='qij' # 版本号
# img5='ianwei/large-lm:1.0.7'
# image="m${img1}.cloud${img2}ioff${img3}/${img4}${img5}"
#img4='wsw/large-lm:1.0.15-3'
img1="icr"
img2=".m"
img3="ice.cn"
#img4='hkx/llm:1.0.2'
# hkx/llm:2.0.0 # nvidia-cuda12.4-python3.11-torch2.6.0-transformers-trl:1.0-multistage-v19
#img4='wsw/large-lm:1.0.15-3'
#img4='wsw/large-lm:1.0.15-2_6'  # torch=2.6
img4='hkx/llm:2.6.0' # torch=2.6
image="m${img1}.cloud${img2}ioff${img3}/${img4}"
echo $image

max_gpu_num=4 # 最大限制多少个gpu
test_max_gpu_num $max_gpu_num
if [ ! -d logs/ ]; then
    mkdir logs/
fi


#wandb_key="bdfc8b674cd322f967699975e89d431e82fcd317" # hkx wandb
port=$(python utils/get_free_port.py)
echo "port:$port"
#torchrun --rdzv-endpoint=localhost:${port} \
#export MASTER_PORT=${port} && \

#port=29501

#pip list|grep -E '(transformers|flash-attn)'; \
# -w: 为工作目录
#pip_index="https://mirrors.aliyun.com/pypi/simple/"
#pip install transformers==4.46.3 -i ${pip_index} && \

<<EOF    
debug:
# micr.cloud.miof
# fice.cn/hkx/llm:1.0.2
docker run -it --gpus 'device=1' -v /etc/localtime:/etc/localtime:ro -v /home/:/home/  --name test_pack_sft --network=host --shm-size=16gb docker_image python3

EOF

# 手动指定gpu
device_list="1,2,3,4,5,6,7"
gpu_num=$(echo ${device_list}|awk -F',' '{print NF}')
set -x
nohup docker run -i --rm --gpus '"device='${device_list}'"'  --name test_grpo --network=host --shm-size=16gb \
    -v /etc/localtime:/etc/localtime:ro \
    -v ${project_path}:/${project_path} \
    -v ${root_path}:/${root_path} \
    -v ${model_out_path}:/${model_out_path} \
    -w ${project_path} \
    ${image} \
    bash -c "\
export PYTHONPATH=/docker_workspace && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 && \
. /opt/conda/etc/profile.d/conda.sh && \
conda activate py_3_10 && \
python train.py --config config_24GB_gpu_docker.yaml \
 " 2>&1 |tee logs/log_${time_str}.txt

set +x
echo "`date` 训练结束"


# torchrun --nproc_per_node=${gpu_num} --rdzv-endpoint=127.0.0.1:${port} \
# train.py --config config_24GB_gpu_docker.yaml \