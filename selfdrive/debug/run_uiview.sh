#!/usr/bin/bash

cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/system/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/data/openpilot/third_party/snpe/aarch64:$LD_LIBRARY_PATH"
export LOGPRINT="debug"
exec ./uiview.py
