#!/bin/bash

# Copies calvin .zip data from /scratch-shared/ to /scratch-node/
# Unzips .zip data into the data folder
# Takes ~ 20 minutes to run

# define source file (passed as arg) and target dir
source_file=$1
target_dir=$TMPDIR/data

# make the target dir
mkdir -p $target_dir

echo "Copying to $target_dir"
cp $source_file $target_dir
echo "Done."

# unzip
echo "Unzipping..."
unzip -q $target_dir/task_D_D.zip -d $target_dir
echo "Done."

$data_dir=$target_dir/task_D_D/
echo "Data files ready in $data_dir"
