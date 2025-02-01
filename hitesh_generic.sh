#!/bin/bash

# Check if a model name was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model_name>"
  echo "Example: $0 llava-onevision-qwen2-0.5b-ov-hf"
  exit 1
fi

# Read the model name from the first command-line argument
MODEL="$1"

# Define your datasets
DATASETS=(
  "AI2D_TEST_sample"
  "ChartQA_TEST_sample"
  "MUIRBench_sample"
  "MMStar_sample"
  "MMMU_TEST_sample"
  "RealWorldQA"
  "MMBench_TEST_EN_V11_sample"
  "GQA_TestDev_Balanced_sample"
  "DocVQA_TEST_sample"
  "AMBER_sample"
  "BLINK_sample"
  "HallusionBench_sample"
  "InfoVQA_TEST_sample"
  "MME_sample"
  "POPE_sample"
  "ScienceQA_TEST_sample"
  "VizWiz_sample"
  "TextVQA_VAL_sample"
)

# Loop through each dataset
for DATA in "${DATASETS[@]}"; do
    echo "Running with dataset: $DATA and model: $MODEL"

    python run.py --data "${DATA}" \
      "${DATA}_grid_2x1_row1_col1" \
      "${DATA}_grid_2x1_row2_col1" \
      "${DATA}_grid_1x2_row1_col1" \
      "${DATA}_grid_1x2_row1_col2" \
      "${DATA}_grid_2x2_row1_col1" \
      "${DATA}_grid_2x2_row1_col2" \
      "${DATA}_grid_2x2_row2_col1" \
      "${DATA}_grid_2x2_row2_col2" \
      "${DATA}_grid_3x3_row1_col1" \
      "${DATA}_grid_3x3_row1_col2" \
      "${DATA}_grid_3x3_row1_col3" \
      "${DATA}_grid_3x3_row2_col1" \
      "${DATA}_grid_3x3_row2_col2" \
      "${DATA}_grid_3x3_row2_col3" \
      "${DATA}_grid_3x3_row3_col1" \
      "${DATA}_grid_3x3_row3_col2" \
      "${DATA}_grid_3x3_row3_col3" \
      --model "${MODEL}" \
      --verbose
done
