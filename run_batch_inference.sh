#!/usr/bin/env bash

set -uo pipefail

MODEL_TARGET='v21-20260903-014339/checkpoint-750'
QUESTION='Key customer service behaviors and states observed in the target users.'
FPS_VALUES=(1.0 3.0 5.0)
WINDOW_SIZE_VALUES=(5 15 25)

DEMO_DIR='demo'
OUTPUT_DIR='output'
PYTHON_BIN="${PYTHON_BIN:-python}"
FFMPEG_BIN="${FFMPEG_BIN:-/usr/bin/ffmpeg}"

if [[ ! -d "$DEMO_DIR" ]]; then
    printf 'Error: directory not found: %s\n' "$DEMO_DIR" >&2
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR/$MODEL_TARGET" ]]; then
    printf 'Error: model path not found: %s\n' "$OUTPUT_DIR/$MODEL_TARGET" >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Error: Python command not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

if ! command -v "$FFMPEG_BIN" >/dev/null 2>&1; then
    printf 'Error: ffmpeg command not found: %s\n' "$FFMPEG_BIN" >&2
    exit 1
fi

if ! "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null \
    | grep '[[:space:]]h264_nvenc[[:space:]]' >/dev/null; then
    printf 'Error: %s does not provide the h264_nvenc encoder.\n' "$FFMPEG_BIN" >&2
    exit 1
fi

mapfile -d '' -t MOVIES < <(
    find "$DEMO_DIR" -mindepth 1 -maxdepth 1 -type f -iname '*.mp4' -print0 | sort -z
)

if (( ${#MOVIES[@]} == 0 )); then
    printf 'Error: no MP4 files found directly under %s.\n' "$DEMO_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

total=$(( ${#MOVIES[@]} * ${#FPS_VALUES[@]} * ${#WINDOW_SIZE_VALUES[@]} ))
current=0
failures=0

for MOVIE_PATH in "${MOVIES[@]}"; do
    MOVIE_FILE=${MOVIE_PATH##*/}
    MOVIE_NAME=${MOVIE_FILE%.*}

    for FPS in "${FPS_VALUES[@]}"; do
        printf -v FPS_PADDED '%02d' "${FPS%%.*}"

        for WINDOW_SIZE in "${WINDOW_SIZE_VALUES[@]}"; do
            printf -v WINDOW_SIZE_PADDED '%02d' "$WINDOW_SIZE"
            ((current += 1))

            OUTPUT_BASE="$OUTPUT_DIR/mtg_${MOVIE_NAME}_${FPS_PADDED}_${WINDOW_SIZE_PADDED}"
            RAW_VIDEO="${OUTPUT_BASE}.raw.mp4"
            COMPRESSED_VIDEO="${OUTPUT_BASE}.mp4"

            printf '\n[%d/%d] movie=%s fps=%s window_size=%s\n' \
                "$current" "$total" "$MOVIE_PATH" "$FPS" "$WINDOW_SIZE"

            if ! "$PYTHON_BIN" inference.py \
                --model-path "$OUTPUT_DIR/$MODEL_TARGET" \
                --video-path "$MOVIE_PATH" \
                --save-video "$RAW_VIDEO" \
                --question "$QUESTION" \
                --fps "$FPS" \
                --window-size "$WINDOW_SIZE"; then
                printf 'Inference failed: %s\n' "$MOVIE_PATH" >&2
                ((failures += 1))
                continue
            fi

            if "$FFMPEG_BIN" -hide_banner -y \
                -i "$RAW_VIDEO" \
                -pix_fmt yuv420p \
                -vcodec h264_nvenc \
                "$COMPRESSED_VIDEO"; then
                rm -f -- "$RAW_VIDEO"
                printf 'Saved: %s\n' "$COMPRESSED_VIDEO"
            else
                printf 'Compression failed; raw video kept at: %s\n' "$RAW_VIDEO" >&2
                ((failures += 1))
            fi
        done
    done
done

if (( failures > 0 )); then
    printf '\nCompleted with %d failed job(s).\n' "$failures" >&2
    exit 1
fi

printf '\nCompleted all %d job(s).\n' "$total"
