#!/bin/bash

# Define your datasets and models
#"VizWiz_sample" "BLINK_sample" "POPE_sample" 
DATASETS=("AI2D_TEST_sample" "ChartQA_TEST_sample" "MUIRBench_sample" "MMStar_sample" "MMMU_TEST_sample")
MODELS=("llava-onevision-qwen2-0.5b-ov-hf" "InternVL2_5-1B" "Phi-3-Vision" "InternVL2-1B")

# Loop through each dataset and model
for DATA in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running with dataset: $DATA and model: $MODEL"
    python run.py --data ${DATA} \
    ${DATA}_grid_2x1_row1_col1 \
    ${DATA}_grid_2x1_row2_col1 \
    ${DATA}_grid_1x2_row1_col1 \
    ${DATA}_grid_1x2_row1_col2 \
    ${DATA}_grid_2x2_row1_col1 \
    ${DATA}_grid_2x2_row1_col2 \
    ${DATA}_grid_2x2_row2_col1 \
    ${DATA}_grid_2x2_row2_col2 \
    ${DATA}_grid_3x3_row1_col1 \
    ${DATA}_grid_3x3_row1_col2 \
    ${DATA}_grid_3x3_row1_col3 \
    ${DATA}_grid_3x3_row2_col1 \
    ${DATA}_grid_3x3_row2_col2 \
    ${DATA}_grid_3x3_row2_col3 \
    ${DATA}_grid_3x3_row3_col1 \
    ${DATA}_grid_3x3_row3_col2 \
    ${DATA}_grid_3x3_row3_col3 \
    --model ${MODEL} --verbose
  done
done
