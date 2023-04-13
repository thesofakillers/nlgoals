#!/bin/bash

# Get the directory path of the script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the default download directory path to the same directory as the script
DOWNLOAD_DIR="$DIR"

# Check if the second argument is provided and set it as the download directory path if it is
if [ -n "$2" ]; then
  DOWNLOAD_DIR="$2"
fi

# Function to print help information
function print_help {
  echo "Usage: download_data.sh D | ABC | ABCD | debug [download directory path]"
  echo "Download and unzip data for the specified task (D, ABC, ABCD, debug)."
  echo "The default download directory is the same directory as the script."
  echo "If a download directory path is provided, the data will be downloaded to that directory instead."
}

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  print_help
  exit 0
fi

# Check if the first argument is provided and download the data accordingly
if [ "$1" = "D" ]; then

  echo "Downloading task_D_D ..."
  wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip -P "$DOWNLOAD_DIR"
  unzip "$DOWNLOAD_DIR/task_D_D.zip" -d "$DOWNLOAD_DIR"
  echo "saved folder: $DOWNLOAD_DIR/task_D_D"

elif [ "$1" = "ABC" ]; then

  echo "Downloading task_ABC_D ..."
  wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip -P "$DOWNLOAD_DIR"
  unzip "$DOWNLOAD_DIR/task_ABC_D.zip" -d "$DOWNLOAD_DIR"
  echo "saved folder: $DOWNLOAD_DIR/task_ABC_D"

elif [ "$1" = "ABCD" ]; then

  echo "Downloading task_ABCD_D ..."
  wget http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip -P "$DOWNLOAD_DIR"
  unzip "$DOWNLOAD_DIR/task_ABCD_D.zip" -d "$DOWNLOAD_DIR"
  echo "saved folder: $DOWNLOAD_DIR/task_ABCD_D"

elif [ "$1" = "debug" ]; then

  echo "Downloading debug dataset ..."
  wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip -P "$DOWNLOAD_DIR"
  unzip "$DOWNLOAD_DIR/calvin_debug_dataset.zip" -d "$DOWNLOAD_DIR"
  echo "saved folder: $DOWNLOAD_DIR/calvin_debug_dataset"

else

  echo "Error: Invalid argument '$1'."
  print_help
  exit 1
fi
