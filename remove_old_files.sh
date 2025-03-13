#!/bin/bash

# 要删除 __pycache__ 文件夹的根目录，默认为当前目录
ROOT_DIR=${1:-.}

# 查找并删除所有的 __pycache__ 文件夹
find "$ROOT_DIR" -type d -name "__pycache__" -exec rm -rf {} +

echo "All __pycache__ directories have been removed from $ROOT_DIR."