#!/bin/bash
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup


# 检查参数数量
if [ $# -ne 4 ]; then
    echo "用法：$0 <log_name> <save_dir> <start_frame> <num_frames>"
    echo "示例：$0 2021.05.12.22.00.38_veh-35_01008_01518 /mnt/data/dataset/nuPlan/processed/01518_frame_0_200 0 200"
    exit 1
fi

# 提取命令行参数
LOG_NAME="$1"
SAVE_DIR="$2"
START_FRAME="$3"
NUM_FRAMES="$4"
NUPLAN_ROOT="/mnt/data/dataset/nuPlan/raw"
SAM_CHECKPOINT="/src/51sim-ai/street_crafter_copy/sky_checkpoint/sam_vit_h_4b8939.pth"
DATADIR="${SAVE_DIR}"

echo "执行 nuplan_converter.py ..."
conda activate streetcrafter && \
python nuplan_converter.py \
    --nuplan_root "${NUPLAN_ROOT}" \
    --log_name "${LOG_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}"

echo "执行 nuplan_get_lidar_pcd.py ..."
conda activate streetcrafter && \
python nuplan_get_lidar_pcd.py \
    --nuplan_root "${NUPLAN_ROOT}" \
    --log_name "${LOG_NAME}" \
    --save_dir "${SAVE_DIR}"

echo "执行 generate_sky_mask_with_8cams.py ..."
conda activate sky_mask_generate && \
python generate_sky_mask_with_8cams.py \
    --datadir "${DATADIR}" \
    --sam_checkpoint "${SAM_CHECKPOINT}"

echo "所有命令执行完毕！"
