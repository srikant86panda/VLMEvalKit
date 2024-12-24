# python run.py --data COCO_VAL_sample COCO_VAL_sample_impulse_noise_1 COCO_VAL_sample_zoom_blur_1 COCO_VAL_sample_snow_1 --model idefics_impulse_noise_coco_dpo_orig_merged --verbose
# python run.py --data COCO_VAL_sample COCO_VAL_sample_impulse_noise_1 --model idefics_impulse_noise_coco_dpo_all_merged --verbose
# python run.py --data COCO_VAL_sample COCO_VAL_sample_impulse_noise_1 --model idefics_impulse_noise_coco_dpo_all_drop_merged --verbose
# python run.py --data COCO_VAL_sample COCO_VAL_sample_impulse_noise_1 --model idefics_impulse_noise_coco_dpo_matching_clean_pred_drop_merged --verbose
# python run.py --data COCO_VAL_sample COCO_VAL_sample_snow_1 --model idefics_snow_coco_dpo_all_merged --verbose
# python run.py --data COCO_VAL_sample COCO_VAL_sample_snow_1 --model idefics_snow_coco_dpo_all_drop_merged --verbose
# python run.py --data COCO_VAL_sample COCO_VAL_sample_snow_1 --model idefics_snow_coco_dpo_matching_clean_pred_drop_merged --verbose
# python run.py --data COCO_VAL_sample COCO_VAL_sample_zoom_blur_1 --model idefics_zoom_coco_dpo_all_merged --verbose
# python run.py --data COCO_VAL_sample COCO_VAL_sample_zoom_blur_1 --model idefics_zoom_coco_dpo_matching_clean_pred_drop_merged --verbose


python run.py --data COCO_VAL_sample COCO_VAL_sample_impulse_noise_1 COCO_VAL_sample_zoom_blur_1 COCO_VAL_sample_snow_1 --model Idefics3-8B-Llama3 llava_hf_v1.5_7b llava_hf_v1.5_13b SmolVLM llava_next_mistral_7b --verbose