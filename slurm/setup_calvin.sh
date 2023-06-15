#!/bin/bash

# Copies calvin .zip data from /scratch-shared/ to /scratch-node/
# Unzips .zip data into the data folder
# Takes ~ 20 minutes to run for task_D_D (~166GB)

# parse source file (passed as arg)
source_path=$1
source_file_name=$(basename $source_path)
source_file_name_no_ext="${source_file_name%.*}"
# define target dir
target_dir=$TMPDIR/data

# make the target dir
mkdir -p $target_dir

echo "Copying to $target_dir"
cp $source_path $target_dir
echo "Done."

# unzip
echo "Unzipping..."
unzip -q $target_dir/$source_file_name -d $target_dir
echo "Done."

$data_dir=$target_dir/$source_file_name_no_ext/
echo "Data files ready in $data_dir"
