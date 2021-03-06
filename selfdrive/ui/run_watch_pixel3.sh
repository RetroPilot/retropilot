#!/bin/sh
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/system/lib64:$LD_LIBRARY_PATH"
export QT_PLUGIN_PATH="../../third_party/qt-plugins/$(uname -m)"
export QT_QPA_EGLFS_HIDECURSOR=1
export QT_QPA_FONTDIR=/usr/share/fonts
export QT_QPA_EVDEV_TOUCHSCREEN_PARAMETERS=/dev/input/event2:rotate=270
export QT_QPA_EGLFS_PHYSICAL_WIDTH=151
export QT_QPA_EGLFS_PHYSICAL_HEIGHT=74
exec ./watch_pixel3
