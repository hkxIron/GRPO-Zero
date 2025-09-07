echo "HOME:$HOME"
# 同步数据
# 在GPU上运行
# 3090, a800上可用
remote_project_path="/home/hkx/data/work/open/GRPO-Zero"
local_project_path_on_gpu_machine="$HOME/work/open/project/"
user_ip="hkx@10.224.104.64"

rsync -av -e ssh --exclude='*.git'  \
--exclude='.*' \
--exclude='*checkpoints*' \
--exclude='__pycache__/' \
--exclude='*.pdf' \
--exclude='*.bin' \
--exclude='*.pt' \
--exclude='wandb/' ${user_ip}:${remote_project_path} $local_project_path_on_gpu_machine
echo "=======================复制数据结束======================"
