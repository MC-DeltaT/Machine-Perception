#!/bin/bash

project_dir=$(dirname "$0")

input_dir="/home/student/train"
output_file="$project_dir/generated/recognition_model.xml"

python3 "$project_dir/src/train_recognition_model.py" "$input_dir" "$output_file"
