#!/usr/bin/env bash
set -euo pipefail

SRC="/home/b920405/git/Streamo/dataset/samples4"
OUT="/home/b920405/git/Streamo/dataset/samples4_archives"
CHUNK_SIZE=3000

mkdir -p "$OUT"
cd "$SRC"

part=1
files=()

create_archive() {
    local archive
    archive=$(printf "%s/samples4_%04d.tar.gz" "$OUT" "$part")

    printf '%s\0' "${files[@]}" |
        tar --null -T - -czf "$archive"

    echo "作成: $archive (${#files[@]} files)"

    files=()
    ((part++))
}

while IFS= read -r -d '' file; do
    files+=("${file#./}")

    if (( ${#files[@]} == CHUNK_SIZE )); then
        create_archive
    fi
done < <(
    find . -type f -print0 |
        sort -z
)

# 3,000個未満の余りを圧縮
if (( ${#files[@]} > 0 )); then
    create_archive
fi