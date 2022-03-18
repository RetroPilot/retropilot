#!/usr/bin/bash

export FINGERPRINT="TOYOTA COROLLA 2010"
export SKIP_FW_QUERY="True"

export PASSIVE="0"
export LD_LIBRARY_PATH="/system/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/data/openpilot/third_party/snpe/aarch64:$LD_LIBRARY_PATH"
export LOGPRINT="debug"
exec ./launch_chffrplus.sh

