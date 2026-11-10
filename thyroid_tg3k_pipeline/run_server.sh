#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
exec 9>.training.lock
flock -n 9 || { echo 'Another pipeline process is running.'; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print(torch.__version__, torch.cuda.get_device_name(0)); x=torch.randn(16,16,device='cuda'); print((x@x).sum().item())"
for path in runs/tg3k_server_v3 runs/local_server_v3 evaluations/test_server_v3; do
    if [ -e "$path" ]; then
        echo "Output already exists: $path. Refusing to overwrite; this script does not resume training."
        exit 1
    fi
done
python verify_server.py
mkdir -p logs
trap 'echo "Pipeline failed at line $LINENO"; exit 1' ERR
python -u train.py --stage pretrain --manifest data/tg3k.csv --run tg3k_server_v3 --device cuda 2>&1 | tee logs/pretrain.log
python -u train.py --stage finetune --init runs/tg3k_server_v3/best.pt --manifest data/original_pairs_v3/local_pairs.csv --run local_server_v3 --device cuda 2>&1 | tee logs/finetune.log
python -u evaluate.py --checkpoint runs/local_server_v3/best.pt --manifest data/original_pairs_v3/final_test.csv --run test_server_v3 --device cuda 2>&1 | tee logs/evaluate.log
printf 'TG3K pretraining, local training and final evaluation completed.\n' > logs/pipeline_complete.txt
