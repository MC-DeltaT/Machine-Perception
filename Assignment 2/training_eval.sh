#!/bin/bash

project_dir=$(dirname "$0")

input_dir="/home/student/train"
model_file="$project_dir/generated/recognition_model.xml"
correct_numbers_file="$project_dir/data/correct_numbers.py"

python3 "$project_dir/src/eval_pipeline.py" "$input_dir" "$model_file" "$correct_numbers_file"
