# export STREAMING_DATASET_PATH="${STREAMING_DATASET_PATH:-./dataset/stream/stream_format.json}"
# export STREAMING_FRAME_DIR="${STREAMING_FRAME_DIR:-./dataset/stream/frames}"
# export STREAMING_DATASET_FPS="${STREAMING_DATASET_FPS:-1.0}"
# export STREAMING_ENABLE_MEMORY_CACHE="${STREAMING_ENABLE_MEMORY_CACHE:-0}"
# export STREAMING_ENABLE_ARCHIVE_RESOLUTION="${STREAMING_ENABLE_ARCHIVE_RESOLUTION:-0}"
# export STREAMING_ARCHIVE_INDEX_PATH="${STREAMING_ARCHIVE_INDEX_PATH:-}"
# export STREAMING_ARCHIVE_GCS_PREFIX="${STREAMING_ARCHIVE_GCS_PREFIX:-}"
# export STREAMING_ARCHIVE_CACHE_DIR="${STREAMING_ARCHIVE_CACHE_DIR:-./dataset/stream/archive_cache}"
# export STREAMING_VIDEO_CACHE_DIR="${STREAMING_VIDEO_CACHE_DIR:-./dataset/stream/video_cache}"

# # Backward-compatible aliases used by earlier archive tooling.
# export STREAM_DATASET_PATH="${STREAM_DATASET_PATH:-$STREAMING_DATASET_PATH}"
# export STREAM_FRAME_CACHE_DIR="${STREAM_FRAME_CACHE_DIR:-$STREAMING_FRAME_DIR}"
# export STREAM_DATASET_FPS="${STREAM_DATASET_FPS:-$STREAMING_DATASET_FPS}"
# export STREAM_ENABLE_MEMORY_CACHE="${STREAM_ENABLE_MEMORY_CACHE:-$STREAMING_ENABLE_MEMORY_CACHE}"
# export STREAM_ENABLE_ARCHIVE_RESOLUTION="${STREAM_ENABLE_ARCHIVE_RESOLUTION:-$STREAMING_ENABLE_ARCHIVE_RESOLUTION}"
# export STREAM_ARCHIVE_INDEX_PATH="${STREAM_ARCHIVE_INDEX_PATH:-$STREAMING_ARCHIVE_INDEX_PATH}"
# export STREAM_ARCHIVE_GCS_PREFIX="${STREAM_ARCHIVE_GCS_PREFIX:-$STREAMING_ARCHIVE_GCS_PREFIX}"
# export STREAM_ARCHIVE_CACHE_DIR="${STREAM_ARCHIVE_CACHE_DIR:-$STREAMING_ARCHIVE_CACHE_DIR}"
# export STREAM_VIDEO_CACHE_DIR="${STREAM_VIDEO_CACHE_DIR:-$STREAMING_VIDEO_CACHE_DIR}"

export MIN_PIXELS=3136
export MAX_PIXELS=100352
export USE_HF=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation
sudo chown -R $(whoami) /mnt/data/downloads/
mkdir -p /mnt/data/downloads/stream
export STREAM_FRAME_CACHE_DIR=/mnt/data/downloads/stream/frames
export TORCH_DISTRIBUTED_DEBUG=DETAIL

### Multi-GPU
# NPROC_PER_NODE=8 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# uv run swift sft \
# --model Qwen2.5-VL-3B-Instruct \
# --dataset streaming_video \
# --train_type full \
# --torch_dtype bfloat16 \
# --num_train_epochs 1 \
# --per_device_train_batch_size 1 \
# --attn_impl flash_attn \
# --padding_free true \
# --loss_type stream \
# --custom_register_path swift/plugin/loss.py swift/plugin/streaming_dataset.py \
# --learning_rate 1e-5 \
# --freeze_vit true \
# --freeze_aligner false \
# --gradient_accumulation_steps 64 \
# --gradient_checkpointing true \
# --save_steps 100 \
# --save_total_limit 2 \
# --logging_steps 5 \
# --output_dir output \
# --warmup_ratio 0.05 \
# --deepspeed zero3 \
# --use_liger_kernel true \
# --max_length 32768 \
# --dataset_num_proc 4 \
# --dataloader_num_workers 4 \
# --new_special_tokens './special_token_v1.txt'

### Single-GPU
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens6
CUDA_VISIBLE_DEVICES=0 \
uv run python swift/cli/sft.py \
--model Qwen/Qwen3-VL-2B-Instruct \
--dataset streaming_video \
--train_type full \
--torch_dtype bfloat16 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--attn_impl flash_attn \
--padding_free true \
--loss_type stream \
--custom_register_path swift/plugin/loss.py swift/plugin/streaming_dataset.py \
--learning_rate 1e-5 \
--freeze_vit true \
--freeze_aligner false \
--gradient_accumulation_steps 64 \
--gradient_checkpointing true \
--save_steps 100 \
--save_total_limit 2 \
--logging_steps 5 \
--output_dir output \
--warmup_ratio 0.05 \
--use_liger_kernel true \
--max_length 32768 \
--dataset_num_proc 4 \
--dataloader_num_workers 4 \
--new_special_tokens './special_token_v1.txt'
