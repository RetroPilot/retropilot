#!/usr/bin/bash

export PASSIVE="0"
export LD_LIBRARY_PATH="/system/lib64:$LD_LIBRARY_PATH"
exec ./launch_chffrplus.sh

