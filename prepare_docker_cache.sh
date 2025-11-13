#!/usr/bin/env bash
set -euo pipefail

# Prepare selected cache artifacts into Docker build context.
#
# 使用方式（不接受命令行参数）：
# 1) 编辑本脚本中的 SOURCES 列表，放入需要复制的“绝对路径”（文件或目录），要求都在 /root/.cache 下。
# 2) 运行脚本： ./prepare_docker_cache.sh
# 3) Dockerfile 会把 .docker-build-cache/ 的全部内容复制到镜像的 /root/.cache/
#
# 约束：
# - SOURCES 列表不能为空；为空则报错退出。
# - 列表中的任意路径不存在或不在 /root/.cache 下，报错退出，不进行任何复制。

CACHE_ROOT_PREFIX="/root/.cache"
DEST_DIR=".docker-build-cache"

# 在此处维护需要复制的绝对路径列表（示例已注释）：
# 例如：
# SOURCES=(
#   "/root/.cache/torch/hub/checkpoints/alex.pth"
#   "/root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
#   "/root/.cache/huggingface/hub"
# )
SOURCES=(
  "/root/.cache/torch/hub/checkpoints/alex.pth"
  "/root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
)

echo "Destination dir: $DEST_DIR"
mkdir -p "$DEST_DIR"

# 校验列表非空
if [[ ${#SOURCES[@]} -eq 0 ]]; then
  echo "Error: SOURCES 列表为空。请在脚本中填写需要复制的绝对路径。" >&2
  exit 1
fi

# 预检：所有路径必须存在且在 /root/.cache 下
missing=0
outside=0
for src in "${SOURCES[@]}"; do
  if [[ ! -e "$src" ]]; then
    echo "Error: 路径不存在: $src" >&2
    missing=1
  fi
  if [[ "$src" != "$CACHE_ROOT_PREFIX"* ]]; then
    echo "Error: 路径不在 $CACHE_ROOT_PREFIX 下: $src" >&2
    outside=1
  fi
done

if [[ $missing -eq 1 || $outside -eq 1 ]]; then
  echo "复制被终止：请修复上述错误后重试。" >&2
  exit 1
fi

for src in "${SOURCES[@]}"; do
  if [[ -z "$src" ]]; then
    continue
  fi
  # 计算相对 /root/.cache 的路径，用于临时目录镜像
  rel="${src#${CACHE_ROOT_PREFIX}/}"

  dst="$DEST_DIR/$rel"

  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    cp -a "$src/." "$dst/"
    echo "Copied dir: $src -> $dst"
  else
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    chmod 644 "$dst" || true
    echo "Copied file: $src -> $dst"
  fi
done

echo "Done. Build context prepared under $DEST_DIR"
echo "Dockerfile will COPY $DEST_DIR/ to /root/.cache inside the image."