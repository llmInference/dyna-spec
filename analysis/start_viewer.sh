#!/bin/bash
# 快速启动Markdown查看器服务器

cd "$(dirname "$0")"
echo "正在启动Markdown查看器服务器..."
echo ""

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 启动服务器
python3 view_server.py "$@"

