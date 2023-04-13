#!/bin/bash

if [ "$1" = "-h" ]; then
  echo "Usage: download_data.sh [-h] [D | ABC | ABCD | debug] [path]"
  echo "Downloads and unzips datasets for tasks D, ABC, ABCD or debug."
  echo "Optional argument [path] specifies the path where to download and unzip the file."
  exit 0
fi

if [ $# -lt 1 ]; then
  echo "Error: Missing argument. Usage download_data.sh [-h] [D | ABC | ABCD | debug] [path]"
  exit 1
fi

if [ "$1" = "D" ]; then
  echo "Downloading task_D_D ..."
  wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip -P "$2"
  unzip -q "$2/task_D_D.zip" -d "$2"
  echo "Saved folder: $2/task_D_D"
elif [ "$1" = "ABC" ]; then
  echo "Downloading task_ABC_D ..."
  wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip -P "$2"
  unzip -q "$2/task_ABC_D.zip" -d "$2"
  echo "Saved folder: $2/task_ABC_D"
elif [ "$1" = "ABCD" ]; then
  echo "Downloading task_ABCD_D ..."
  wget http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip -P "$2"
  unzip -q "$2/task_ABCD_D.zip" -d "$2"
  echo "Saved folder: $2/task_ABCD_D"
elif [ "$1" = "debug" ]; then
  echo "Downloading debug dataset ..."
  wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip -P "$2"
  unzip -q "$2/calvin_debug_dataset.zip" -d "$2"
  echo "Saved folder: $2/calvin_debug_dataset"
else
  echo "Error: Invalid argument. Usage download_data.sh [-h] [D | ABC | ABCD | debug] [path]"
  exit 1
fi
