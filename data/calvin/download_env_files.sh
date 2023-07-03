#!/bin/bash

# Usage: bash download_env_files.sh -o <output_dir>
# Downloads the calvin_env repository to <output_dir>/calvin_env

# set default output dir to current dir
output_dir=$pwd

# help message
usage() {
  echo "Usage: $0 [-o/--output <output_dir>]"
  echo "Downloads the calvin_env repository"
  echo ""
  echo "Options:"
  echo "  -o/--output       Specify output directory (default: current directory)"
  echo "  -h                Display this help message"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o | --output)
      output_dir="${2}"
      shift 2
      ;;
    -h | --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done


target_dir=$output_dir/calvin_env
# make target directory
mkdir -p $target_dir

# download calvin_env repository
git clone git@github.com:thesofakillers/calvin_env.git $target_dir
