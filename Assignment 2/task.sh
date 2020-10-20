#!/bin/bash

project_dir=$(dirname "$0")

input_dir="/home/student/test"
output_dir="$project_dir/output"
model_file="$project_dir/generated/recognition_model.xml"

rm -rf "$output_dir"
python3 "$project_dir/src/extract_house_numbers.py" "$input_dir" "$model_file" "$output_dir"
