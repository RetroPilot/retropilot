#!/bin/bash
#==============================================================================
#
#  Copyright (c) 2016-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

# This script checks if the python dependencies are met

PYV=`python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";`
if [ $PYV = '3.6' ]; then
    echo Supported version of Python found: $PYV
    # Check if there are multiple versions of python modules and warn user
    #Dependencies that are needed for running snpe
    needed_depends=()
    needed_depends+=('python3-numpy')
    needed_depends+=('python-sphinx')
    needed_depends+=('python3-scipy')
    needed_depends+=('python3-matplotlib')
    needed_depends+=('python3-skimage')
    needed_depends+=('python-protobuf')
    needed_depends+=('python3-yaml')
    needed_depends+=('python3-mako')

    #Dependencies that are needed for running snpe
    needed_depends_pip=()
    needed_depends_pip+=('numpy')
    needed_depends_pip+=('sphinx')
    needed_depends_pip+=('scipy')
    needed_depends_pip+=('matplotlib')
    needed_depends_pip+=('scikit-image')
    needed_depends_pip+=('protobuf')
    needed_depends_pip+=('pyyaml')
    needed_depends_pip+=('mako')

    #Unmet dependencies
    need_to_install=()

    #Check if pip is installed
    PIP3_INSTALLED=false
    if type pip3 &> /dev/null; then
        PIP3_INSTALLED=true
    fi

    i=0
    while [ $i -lt ${#needed_depends[*]} ]; do

      PKG_INSTALLED=$(dpkg-query -W --showformat='${Status}\n' ${needed_depends[$i]}|grep "install ok installed")
      echo "Checking for ${needed_depends[$i]}: $PKG_INSTALLED"
      if [ "$PKG_INSTALLED" != "" ]; then
          if [ "$PIP3_INSTALLED" = "true" ]; then
              pip_version_str=$(pip3 show ${needed_depends_pip[$i]} | grep "Version")
              if [[ ! -z "$pip_version_str" ]]; then
                  echo "WARNING: It appears the python module ${needed_depends_pip[$i]} is installed on this system using the apt-get distribution as well as pip. If you encounter errors, please use only one distribution."
              fi
          fi
      elif [ "$PIP3_INSTALLED" = "true" ]; then
          pip_version_str=$(pip3 show ${needed_depends_pip[$i]} | grep "Version")
          if [[ ! -z "$pip_version_str" ]]; then
                echo "${needed_depends_pip[$i]} installed via pip ${pip_version_str}"
          fi
      fi
      echo "==========================================="
      i=$(( $i +1));
    done
else
    echo Supported versions of Python are 3.6 . Found instead:  $PYV
fi

