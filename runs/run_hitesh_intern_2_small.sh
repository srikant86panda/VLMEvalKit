#!/bin/bash

# Define your datasets and models
#"VizWiz_sample" "BLINK_sample" "POPE_sample" 
DATASETS=("AI2D_TEST_sample" "ChartQA_TEST_sample" "MUIRBench_sample" "MMStar_sample" "MMMU_TEST_sample" "RealWorldQA" "MMBench_TEST_EN_V11_sample" "GQA_TestDev_Balanced_sample" "DocVQA_TEST_sample" "AMBER_sample" "BLINK_sample" "HallusionBench_sample"
"InfoVQA_TEST_sample" "MME_sample" "POPE_sample" "ScienceQA_TEST_sample" "VizWiz_sample" "TextVQA_VAL_sample")
MODELS=("InternVL2_5-1B" "InternVL2_5-2B")

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
