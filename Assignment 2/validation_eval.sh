#!/bin/bash

project_dir=$(dirname "$0")

input_dir="/home/student/val"
model_file="$project_dir/generated/recognition_model.xml"
correct_numbers_file="$project_dir/data/correct_numbers.py"

python3 eval_pipeline.py "$input_dir" "$model_file" "$correct_numbers_file"
