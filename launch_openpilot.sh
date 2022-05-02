#!/usr/bin/bash

export PASSIVE="0"
export LD_LIBRARY_PATH="/system/lib64:$LD_LIBRARY_PATH"
export LOGPRINT="debug"
exec ./launch_chffrplus.sh

