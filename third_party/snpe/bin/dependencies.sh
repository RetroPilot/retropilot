#! /bin/bash
#==============================================================================
#
#  Copyright (c) 2016,2020,2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

#Dependencies that are needed for sdk running
needed_depends=()
needed_depends+=('python3-dev')
needed_depends+=('wget')
needed_depends+=('zip')
needed_depends+=('libc++-9-dev')

#number of version_depends must match number of needed_depends
version_depends=()
version_depends+=('Version: 3.6.7-1~18.04')
version_depends+=('Version: 1.19.4-1ubuntu2.2')
version_depends+=('Version: 3.0-11build1')
version_depends+=('Version: 1:9-2~ubuntu18.04.2')

#Unmet dependencies
need_to_install=()

i=0
while [ $i -lt ${#needed_depends[*]} ]; do
  PKG_INSTALLED=$(dpkg-query -W --showformat='${Status}\n' ${needed_depends[$i]}|grep "install ok installed")
  echo "Checking for ${needed_depends[$i]}: $PKG_INSTALLED"
  if [ "$PKG_INSTALLED" == "" ]; then
      echo "${needed_depends[$i]} is not installed. Adding to list of packages to be installed"
      need_to_install+=(${needed_depends[$i]})
  else
      current_version=$(dpkg -s ${needed_depends[$i]} | grep Version)
      if [ "$current_version" == "${version_depends[$i]}" ]; then
          echo "Success: Version of ${needed_depends[$i]} matches tested version"
      else
          echo "WARNING: Version of ${needed_depends[$i]} on this system which is $current_version does not match tested version which is ${version_depends[$i]}"
      fi
  fi
  i=$(( $i +1));
done

for j in "${need_to_install[@]}"
do
    sudo apt-get install $j
done


