#!/usr/bin/env bash

# 設定環境變數
export sif_image=/home/bill/Documents/research/CVPR_gembench_baseline/nvcuda_v2.sif
export python_bin=$HOME/anaconda3/envs/gembench/bin/python
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

seed=200
data_dir=data/gembench
img_size=128
num_episodes=10

# 載入 task+variation 的 json 檔
taskvars_json=/home/bill/Documents/research/CVPR_gembench_baseline/robot-3dlotus/assets/taskvars_train.json

# 檢查 jq 是否安裝
if ! command -v jq &> /dev/null; then
  echo "請先安裝 jq (e.g. sudo apt-get install jq)"
  exit 1
fi

# 迴圈跑每一組 task+variation
jq -r '.[]' "${taskvars_json}" | while IFS= read -r tv; do
  # 分割字串 "push_button+3" → task="push_button" variation="3"
  task="${tv%%+*}"
  variation="${tv##*+}"

  echo "==== 處理任務: ${task} 變化: ${variation} ===="

  microstep_data_dir="${data_dir}/myRLBench_train_dataset/microsteps/seed${seed}"

  singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv "${sif_image}" \
    xvfb-run -a "${python_bin}" preprocess/generate_dataset_microsteps.py \
      --microstep_data_dir "${microstep_data_dir}" \
      --task "${task}" \
      --variation_id "${variation}" \
      --seed "${seed}" \
      --image_size "${img_size}" \
      --renderer opengl \
      --episodes_per_task "${num_episodes}" \
      --live_demos

done

echo "全部任務跑完！"