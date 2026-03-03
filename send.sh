#! /bin/zsh
uv tool run ruff format
# Get list of files updated since last commit
#files=`git diff --name-only HEAD`
files=("${(@f)$(git --no-pager diff --name-only HEAD)}")

# Specify the destination host and folder
dir=$(pwd)
host="cluster"
folder="/cluster/research-groups/wehrwein/home/finn/Tome/"

# Send each file to the host using SCP
for file in $files; do
    echo sending: $file to: $host:$folder$file
    scp $file $host:$folder$file
done

# scp -qr ./training/experiments/ $host:${folder}training/
