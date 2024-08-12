
Task list:
MMMU_TEST MME TextVQA_VAL POPE OCRBench ScienceQA_TEST BLINK COCO_VAL RealWorldQA ChartQA_TEST OCRVQA_TEST

Model list:
Env 1: InternVL2-1B InternVL2-2B InternVL2-4B InternVL2-8B InternVL2-26B InternVL2-40B InternVL2-76B llava_v1.5_7b llava_v1.5_13b
Env 2: Phi-3-Vision
Env 3: chameleon_7b chameleon_30b


CUDA_VISIBLE_DEVICES=0 python run.py --data RealWorldQA --model paligemma-3b-mix-448 --verbose
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 run.py --data RealWorldQA --model paligemma-3b-mix-448 --verbose