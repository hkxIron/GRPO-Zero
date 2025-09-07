import sys

import torch

def get_free_gpu_ids(max_gpu_num:int):
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        if torch.cuda.memory_allocated(i) == 0 and torch.cuda.memory_reserved(i) == 0:
            if len(available_gpus)<max_gpu_num:
                available_gpus.append(str(i))
    return available_gpus

gpus = get_free_gpu_ids(int(sys.argv[1]))
if len(gpus)>0:
    print(",".join(gpus))