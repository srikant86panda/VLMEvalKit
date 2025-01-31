import warnings

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
import pandas as pd

MMMB_URLS = {
    'MMMB_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ar.tsv',
    'MMMB_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_cn.tsv',
    'MMMB_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_en.tsv',
    'MMMB_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_pt.tsv',
    'MMMB_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ru.tsv',
    'MMMB_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_tr.tsv',
}

MTL_MMBench_URLS = {
    'MMBench_dev_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ar.tsv',
    'MMBench_dev_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_cn.tsv',
    'MMBench_dev_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_en.tsv',
    'MMBench_dev_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_pt.tsv',
    'MMBench_dev_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_tr.tsv',
    'MMBench_dev_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ru.tsv',
}

MMMB_MD5 = {
    'MMMB_ar': 'f3a18b6385f1d9701840aa42de27aead', 'MMMB_cn': '13ed82fa89730037292fcaa27f08f430',
    'MMMB_en': '1cd781a71ec5a2983c090b84105d6a01', 'MMMB_pt': '548ea2b3bb2da991790386f0015d30d1',
    'MMMB_ru': 'ce1cc8a0533425ab0d86b326ebfc2984', 'MMMB_tr': '0733739d43090327975294292bc5cd67'
}

MTL_MMBench_MD5 = {
    'MMBench_dev_ar': '4271b4a0d0200e1a86380a878e0d64a4', 'MMBench_dev_cn': '2ed5135326fed02c8e51ea50dda8222f',
    'MMBench_dev_en': 'd9ab776fc018b3d45785e9a5c23431c2', 'MMBench_dev_pt': '4ddfbcd27ef12444b908c03831cd0295',
    'MMBench_dev_tr': '4fab39d501389d3d6cc90264bb708f11', 'MMBench_dev_ru': '5ba1171ff2e68f80637bf78349e402a5'
}


class ImageMCQDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        # MMBench v1.0
        'MMBench_DEV_EN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv',
        'MMBench_TEST_EN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN.tsv',
        'MMBench_DEV_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_CN.tsv',
        'MMBench_TEST_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_CN.tsv',
        'MMBench': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench.tsv',  # Internal
        'MMBench_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_CN.tsv',  # Internal
        # MMBench v1.1
        'MMBench_DEV_EN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv',
        'MMBench_TEST_EN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN_V11.tsv',
        'MMBench_DEV_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_CN_V11.tsv',
        'MMBench_TEST_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_CN_V11.tsv',
        'MMBench_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_V11.tsv',  # Internal
        'MMBench_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_CN_V11.tsv',  # Internal
        'MMBench_TEST_EN_V11_sample' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_2x1_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_2x1_row1_col1.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_2x1_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_2x1_row2_col1.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_1x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_1x2_row1_col1.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_1x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_1x2_row1_col2.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_2x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_2x2_row1_col1.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_2x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_2x2_row1_col2.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_2x2_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_2x2_row2_col1.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_2x2_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_2x2_row2_col2.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_3x3_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_3x3_row1_col1.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_3x3_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_3x3_row1_col2.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_3x3_row1_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_3x3_row1_col3.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_3x3_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_3x3_row2_col1.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_3x3_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_3x3_row2_col2.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_3x3_row2_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_3x3_row2_col3.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_3x3_row3_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_3x3_row3_col1.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_3x3_row3_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_3x3_row3_col2.tsv'
        ),
        'MMBench_TEST_EN_V11_sample_grid_3x3_row3_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMBench_TEST_EN_V11/MMBench_TEST_EN_V11_sample_grid_3x3_row3_col3.tsv'
        ),
        # SEEDBench Series
        'SEEDBench_IMG': 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench_IMG.tsv',
        'SEEDBench2': 'https://huggingface.co/datasets/VLMEval/SEEDBench2/resolve/main/SEEDBench2.tsv',
        'SEEDBench2_Plus': 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench2_Plus.tsv',
        # ScienceQA Series
        'ScienceQA_VAL': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_VAL.tsv',
        'ScienceQA_TEST': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_TEST.tsv',
        'ScienceQA_TEST_sample' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/resolve/main/'
            'ScienceQA_TEST/ScienceQA_TEST_sample.tsv'
        ),
        'ScienceQA_TEST_samsple_grid_2x1_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_2x1_row1_col1.tsv'
        ),
        'ScienceQA_TEST_sample_grid_2x1_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_2x1_row2_col1.tsv'
        ),
        'ScienceQA_TEST_sample_grid_1x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_1x2_row1_col1.tsv'
        ),
        'ScienceQA_TEST_sample_grid_1x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_1x2_row1_col2.tsv'
        ),
        'ScienceQA_TEST_sample_grid_2x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_2x2_row1_col1.tsv'
        ),
        'ScienceQA_TEST_sample_grid_2x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_2x2_row1_col2.tsv'
        ),
        'ScienceQA_TEST_sample_grid_2x2_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_2x2_row2_col1.tsv'
        ),
        'ScienceQA_TEST_sample_grid_2x2_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_2x2_row2_col2.tsv'
        ),
        'ScienceQA_TEST_sample_grid_3x3_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_3x3_row1_col1.tsv'
        ),
        'ScienceQA_TEST_sample_grid_3x3_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_3x3_row1_col2.tsv'
        ),
        'ScienceQA_TEST_sample_grid_3x3_row1_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_3x3_row1_col3.tsv'
        ),
        'ScienceQA_TEST_sample_grid_3x3_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_3x3_row2_col1.tsv'
        ),
        'ScienceQA_TEST_sample_grid_3x3_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_3x3_row2_col2.tsv'
        ),
        'ScienceQA_TEST_sample_grid_3x3_row2_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_3x3_row2_col3.tsv'
        ),
        'ScienceQA_TEST_sample_grid_3x3_row3_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_3x3_row3_col1.tsv'
        ),
        'ScienceQA_TEST_sample_grid_3x3_row3_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_3x3_row3_col2.tsv'
        ),
        'ScienceQA_TEST_sample_grid_3x3_row3_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/ScienceQA_TEST/ScienceQA_TEST_sample_grid_3x3_row3_col3.tsv'
        ),     
        # MMT-Bench
        'MMT-Bench_ALL_MI': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_ALL_MI.tsv',
        'MMT-Bench_ALL': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_ALL.tsv',
        'MMT-Bench_VAL_MI': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL_MI.tsv',
        'MMT-Bench_VAL': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL.tsv',
        # AesBench
        'AesBench_VAL': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_VAL.tsv',
        'AesBench_TEST': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_TEST.tsv',
        # Q-Bench1
        'Q-Bench1_VAL': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_VAL.tsv',
        'Q-Bench1_TEST': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_TEST.tsv',
        # A-Bench
        'A-Bench_VAL': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_VAL.tsv',
        'A-Bench_TEST': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_TEST.tsv',
        # R-Bench
        'R-Bench-Dis': 'https://huggingface.co/datasets/lcysyzxdxc/R-Bench/blob/main/R-bench-dis.tsv',
        'R-Bench-Ref': 'https://huggingface.co/datasets/lcysyzxdxc/R-Bench/blob/main/R-bench-ref.tsv',
        # Other Benchmarks
        'CCBench': 'https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv',
        # AI2D_TEST & AI2D_TEST_samples
        'AI2D_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv',
        'AI2D_TEST_sample' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/resolve/main/'
            'AI2D_TEST/AI2D_TEST_sample.tsv'
        ),
        'AI2D_TEST_sample_grid_2x1_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_2x1_row1_col1.tsv'
        ),
        'AI2D_TEST_sample_grid_2x1_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_2x1_row2_col1.tsv'
        ),
        'AI2D_TEST_sample_grid_1x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_1x2_row1_col1.tsv'
        ),
        'AI2D_TEST_sample_grid_1x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_1x2_row1_col2.tsv'
        ),
        'AI2D_TEST_sample_grid_2x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_2x2_row1_col1.tsv'
        ),
        'AI2D_TEST_sample_grid_2x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_2x2_row1_col2.tsv'
        ),
        'AI2D_TEST_sample_grid_2x2_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_2x2_row2_col1.tsv'
        ),
        'AI2D_TEST_sample_grid_2x2_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_2x2_row2_col2.tsv'
        ),
        'AI2D_TEST_sample_grid_3x3_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_3x3_row1_col1.tsv'
        ),
        'AI2D_TEST_sample_grid_3x3_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_3x3_row1_col2.tsv'
        ),
        'AI2D_TEST_sample_grid_3x3_row1_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_3x3_row1_col3.tsv'
        ),
        'AI2D_TEST_sample_grid_3x3_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_3x3_row2_col1.tsv'
        ),
        'AI2D_TEST_sample_grid_3x3_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_3x3_row2_col2.tsv'
        ),
        'AI2D_TEST_sample_grid_3x3_row2_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_3x3_row2_col3.tsv'
        ),
        'AI2D_TEST_sample_grid_3x3_row3_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_3x3_row3_col1.tsv'
        ),
        'AI2D_TEST_sample_grid_3x3_row3_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_3x3_row3_col2.tsv'
        ),
        'AI2D_TEST_sample_grid_3x3_row3_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AI2D_TEST/AI2D_TEST_sample_grid_3x3_row3_col3.tsv'
        ),     
        'AI2D_TEST_NO_MASK': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST_NO_MASK.tsv',
        # MMstar and MMstar sample
        'MMStar': 'https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv',
        'MMStar_sample' : 'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/resolve/main/MMStar/MMStar_sample.tsv',
        'MMStar_sample_grid_2x1_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_2x1_row1_col1.tsv'
        ),
        'MMStar_sample_grid_2x1_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_2x1_row2_col1.tsv'
        ),
        'MMStar_sample_grid_1x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_1x2_row1_col1.tsv'
        ),
        'MMStar_sample_grid_1x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_1x2_row1_col2.tsv'
        ),
        'MMStar_sample_grid_2x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_2x2_row1_col1.tsv'
        ),
        'MMStar_sample_grid_2x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_2x2_row1_col2.tsv'
        ),
        'MMStar_sample_grid_2x2_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_2x2_row2_col1.tsv'
        ),
        'MMStar_sample_grid_2x2_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_2x2_row2_col2.tsv'
        ),
        'MMStar_sample_grid_3x3_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_3x3_row1_col1.tsv'
        ),
        'MMStar_sample_grid_3x3_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_3x3_row1_col2.tsv'
        ),
        'MMStar_sample_grid_3x3_row1_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_3x3_row1_col3.tsv'
        ),
        'MMStar_sample_grid_3x3_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_3x3_row2_col1.tsv'
        ),
        'MMStar_sample_grid_3x3_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_3x3_row2_col2.tsv'
        ),
        'MMStar_sample_grid_3x3_row2_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_3x3_row2_col3.tsv'
        ),
        'MMStar_sample_grid_3x3_row3_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_3x3_row3_col1.tsv'
        ),
        'MMStar_sample_grid_3x3_row3_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_3x3_row3_col2.tsv'
        ),
        'MMStar_sample_grid_3x3_row3_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMStar/MMStar_sample_grid_3x3_row3_col3.tsv'
        ), 
        # RealWorldQA and RealWorldQA sample
        'RealWorldQA': 'https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv',
        'RealWorldQA_grid_2x1_row1_col1' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_2x1_row1_col1.tsv'
        ),
        'RealWorldQA_grid_2x1_row2_col1' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_2x1_row2_col1.tsv'
        ),
        'RealWorldQA_grid_1x2_row1_col1' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_1x2_row1_col1.tsv'
        ),
        'RealWorldQA_grid_1x2_row1_col2' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_1x2_row1_col2.tsv'
        ),
        'RealWorldQA_grid_2x2_row1_col1' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_2x2_row1_col1.tsv'
        ),
        'RealWorldQA_grid_2x2_row1_col2' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_2x2_row1_col2.tsv'
        ),
        'RealWorldQA_grid_2x2_row2_col1' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_2x2_row2_col1.tsv'
        ),
        'RealWorldQA_grid_2x2_row2_col2' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_2x2_row2_col2.tsv'
        ),
        'RealWorldQA_grid_3x3_row1_col1' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_3x3_row1_col1.tsv'
        ),
        'RealWorldQA_grid_3x3_row1_col2' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_3x3_row1_col2.tsv'
        ),
        'RealWorldQA_grid_3x3_row1_col3' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_3x3_row1_col3.tsv'
        ),
        'RealWorldQA_grid_3x3_row2_col1' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_3x3_row2_col1.tsv'
        ),
        'RealWorldQA_grid_3x3_row2_col2' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_3x3_row2_col2.tsv'
        ),
        'RealWorldQA_grid_3x3_row2_col3' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_3x3_row2_col3.tsv'
        ),
        'RealWorldQA_grid_3x3_row3_col1' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_3x3_row3_col1.tsv'
        ),
        'RealWorldQA_grid_3x3_row3_col2' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_3x3_row3_col2.tsv'
        ),
        'RealWorldQA_grid_3x3_row3_col3' : (
            'https://huggingface.co/datasets/Srikant86/VLMEval/'
            'resolve/main/RealWorldQA/RealWorldQA_grid_3x3_row3_col3.tsv'
        ),
        'MLLMGuard_DS': 'https://opencompass.openxlab.space/utils/VLMEval/MLLMGuard_DS.tsv',
        'BLINK': 'https://opencompass.openxlab.space/utils/VLMEval/BLINK.tsv',
        'BLINK_sample' : 'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/resolve/main/BLINK/BLINK_sample.tsv',
        'BLINK_sample_grid_2x1_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_2x1_row1_col1.tsv'
        ),
        'BLINK_sample_grid_2x1_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_2x1_row2_col1.tsv'
        ),
        'BLINK_sample_grid_1x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_1x2_row1_col1.tsv'
        ),
        'BLINK_sample_grid_1x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_1x2_row1_col2.tsv'
        ),
        'BLINK_sample_grid_2x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_2x2_row1_col1.tsv'
        ),
        'BLINK_sample_grid_2x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_2x2_row1_col2.tsv'
        ),
        'BLINK_sample_grid_2x2_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_2x2_row2_col1.tsv'
        ),
        'BLINK_sample_grid_2x2_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_2x2_row2_col2.tsv'
        ),
        'BLINK_sample_grid_3x3_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_3x3_row1_col1.tsv'
        ),
        'BLINK_sample_grid_3x3_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_3x3_row1_col2.tsv'
        ),
        'BLINK_sample_grid_3x3_row1_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_3x3_row1_col3.tsv'
        ),
        'BLINK_sample_grid_3x3_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_3x3_row2_col1.tsv'
        ),
        'BLINK_sample_grid_3x3_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_3x3_row2_col2.tsv'
        ),
        'BLINK_sample_grid_3x3_row2_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_3x3_row2_col3.tsv'
        ),
        'BLINK_sample_grid_3x3_row3_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_3x3_row3_col1.tsv'
        ),
        'BLINK_sample_grid_3x3_row3_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_3x3_row3_col2.tsv'
        ),
        'BLINK_sample_grid_3x3_row3_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/BLINK/BLINK_sample_grid_3x3_row3_col3.tsv'                                         
        ),
        'TaskMeAnything_v1_imageqa_random': (
            'https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random/'
            'resolve/main/TaskMeAnything-v1-imageqa-random.tsv'
        ),
        'A-OKVQA': 'https://huggingface.co/datasets/Allen8/A-OKVQA/resolve/main/a-okvqa.tsv',
        'WorldMedQA-V': 'https://opencompass.openxlab.space/utils/VLMEval/WorldMedQA-V.tsv',
        'VisOnlyQA-VLMEvalKit': (
            'https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real/'
            'resolve/main/visonlyqa_vlmevalkit.tsv'
        ),
    }

    DATASET_MD5 = {
        # MMBench v1.0
        'MMBench_DEV_EN': 'b6caf1133a01c6bb705cf753bb527ed8',
        'MMBench_TEST_EN': '6939fadb0ce626fefc0bdc9c64efc528',
        'MMBench_DEV_CN': '08b8fc3324a5ed74155350f57be69fbd',
        'MMBench_TEST_CN': '7e1239baf0ee4c8b513e19705a0f317e',
        'MMBench': '4115aea3383f3dd0083be6a633e0f820',  # Internal Only
        'MMBench_CN': '2e053ffc90ea598b1feae13c36dc13ee',    # Internal Only
        # MMBench v1.1
        'MMBench_DEV_EN_V11': '30c05be8f2f347a50be25aa067248184',
        'MMBench_TEST_EN_V11': '26f0f15381a21720255091d3e0316ce6',
        'MMBench_DEV_CN_V11': '593f9b5f6bea453d870a798b34ae4f37',
        'MMBench_TEST_CN_V11': '74bbe4556dac745613c7cbe5ad787050',
        'MMBench_V11': 'b9276414f57af1308dcc4d0cd9b42e7c',  # Internal Only
        'MMBench_TEST_EN_V11_sample': 'c9ecfe4f2bbff61f50172c6a08a8f430',    # Internal Only
        'MMBench_TEST_EN_V11_sample_grid_2x1_row1_col1': '5be57a5ef1add2f36c5d18ffdb4ecb99',
        'MMBench_TEST_EN_V11_sample_grid_2x1_row2_col1': '51901c2bbbc6ba7830f42c7dda6caad2',
        'MMBench_TEST_EN_V11_sample_grid_1x2_row1_col1': 'fc3fedeb47ccf2034b76bfa721b5d2b3',
        'MMBench_TEST_EN_V11_sample_grid_1x2_row1_col2': '0d7ed5c89579d3a6fda28e20486161f2',
        'MMBench_TEST_EN_V11_sample_grid_2x2_row1_col1': '5464823c901120e11eebc5f98b4f6521',
        'MMBench_TEST_EN_V11_sample_grid_2x2_row1_col2': 'eb9bec006208fa8f7ad3ad7f921b71ca',
        'MMBench_TEST_EN_V11_sample_grid_2x2_row2_col1': '5dcae6df9ddb71a8042e68780a6d0117',
        'MMBench_TEST_EN_V11_sample_grid_2x2_row2_col2': 'bfa20c21f964515fc54b0cf8257e3af3',
        'MMBench_TEST_EN_V11_sample_grid_3x3_row1_col1': 'c56877dfefd0c0a36e43aef09da46685',
        'MMBench_TEST_EN_V11_sample_grid_3x3_row1_col2': '78a4ec926b75452e1765c623dfab8309',
        'MMBench_TEST_EN_V11_sample_grid_3x3_row1_col3': 'f42e799baa9019facd330adf40bf1002',
        'MMBench_TEST_EN_V11_sample_grid_3x3_row2_col1': 'ca5c8cff77d840565b11354cc4f36a11',
        'MMBench_TEST_EN_V11_sample_grid_3x3_row2_col2': '3426e2581356d74f9d29ec41225318bc',
        'MMBench_TEST_EN_V11_sample_grid_3x3_row2_col3': '254e102ecaa25465cea23deefe02b07b',
        'MMBench_TEST_EN_V11_sample_grid_3x3_row3_col1': 'd3240d1e17155d7e2d8377117bba4b62',
        'MMBench_TEST_EN_V11_sample_grid_3x3_row3_col2': '73a68a18a28ea93dbfdb7badeaffb7af',
        'MMBench_TEST_EN_V11_sample_grid_3x3_row3_col3': '6cbd8d764077f2ca897571fad86bdb1b',
        # SEEDBench
        'SEEDBench_IMG': '68017231464752261a2526d6ca3a10c0',
        'SEEDBench2': '4ec15cf864c4f16274112284f531813e',
        'SEEDBench2_Plus': 'e32d3216dc4f452b0fe497a52015d1fd',
        # ScienceQA
        'ScienceQA_VAL': '96320d05e142e585e7204e72affd29f3',
        'ScienceQA_TEST': 'e42e9e00f9c59a80d8a5db35bc32b71f',
        'ScienceQA_TEST_sample': '0a6011503051fbad8cc0de2ee63b66f8',
        'ScienceQA_TEST_sample_grid_2x1_row1_col1': '55aac9fbc1776dc0c4cfb4dc8987d884',
        'ScienceQA_TEST_sample_grid_2x1_row2_col1': '5e883b73ba711dca4c091be9a85aca7d',
        'ScienceQA_TEST_sample_grid_1x2_row1_col1': '590c96865bf1641ab913f29645be7c24',
        'ScienceQA_TEST_sample_grid_1x2_row1_col2': '81bfc1a159115a05f731528058e952e2',
        'ScienceQA_TEST_sample_grid_2x2_row1_col1': 'e895da89c0cc0acf74dbbb043ba18014',
        'ScienceQA_TEST_sample_grid_2x2_row1_col2': '7abc229ffd24605040d269f7548f3bd1',
        'ScienceQA_TEST_sample_grid_2x2_row2_col1': '495b8330c529fdb7e8175ac819f30754',
        'ScienceQA_TEST_sample_grid_2x2_row2_col2': 'd3fab0e507ddad3b9f0a3497beb0a2a9',
        'ScienceQA_TEST_sample_grid_3x3_row1_col1': '4c3d4d1f6c9ee72ea1f302f6216af85a',
        'ScienceQA_TEST_sample_grid_3x3_row1_col2': 'c7251c260da0db6e834c50d51382d3e5',
        'ScienceQA_TEST_sample_grid_3x3_row1_col3': '7a7bb286a56a21a8a80079befa3d0af3',
        'ScienceQA_TEST_sample_grid_3x3_row2_col1': '38f64feb2e00e0dfaa4a33344cab9625',
        'ScienceQA_TEST_sample_grid_3x3_row2_col2': 'c21e8c173ebdaadba1913986d5780f36',
        'ScienceQA_TEST_sample_grid_3x3_row2_col3': 'c305fa3dba70195a6b4a3193a77a59c5',
        'ScienceQA_TEST_sample_grid_3x3_row3_col1': '0addfc67866cf2fa954025a7c3fa9f22',
        'ScienceQA_TEST_sample_grid_3x3_row3_col2': '9456e4ada90959125222211cc142c07d',
        'ScienceQA_TEST_sample_grid_3x3_row3_col3': '532eb7be93598e6f4e318d1f3b2bbd0d', 
        # MMT-Bench
        'MMT-Bench_ALL_MI': '5272157097e19cdd7cb41e412ab3b7c7',
        'MMT-Bench_ALL': 'b273a2f4c596fe4f2605de0494cd632f',
        'MMT-Bench_VAL_MI': 'c7d7b998eb5cd9aa36c7d4f721472462',
        'MMT-Bench_VAL': '8dd4b730f53dbf9c3aed90ca31c928e0',
        # AesBench
        'AesBench_VAL': '3edb0c319e9187aa0b97fe7a11700a8c',
        'AesBench_TEST': '58b1f7ba2cc32e1d68896d6ee716bbf8',
        # Q-Bench1
        'Q-Bench1_VAL': '837bdb6cd2da571713543462815187b7',
        'Q-Bench1_TEST': '15e759bfd58c9d5f30b23a317d347153',
        # A-Bench
        'A-Bench_VAL': '218563ec50d34bb336c814143a5bb9c1',
        'A-Bench_TEST': '567013fb033a20cf23f51d8e865bd16c',
        # R-Bench
        'R-Bench-Dis': 'd6e961dbfc43350688af2560226830b4',
        'R-Bench-Ref': '270c1cb555acb523f3fdb178ed57021d',
        # Other Benchmarks
        'CCBench': 'f5dde47f24dc5a6fb6e595b409b466ac',
        # AI2D_TEST
        'AI2D_TEST': '0f593e0d1c7df9a3d69bf1f947e71975',
        'AI2D_TEST_sample': 'de42b4155d4afbc490d2cd7f7a765066',
        'AI2D_TEST_sample_grid_2x1_row1_col1': 'ac3181a094a1a32fbc2014f48cffdd1b',
        'AI2D_TEST_sample_grid_2x1_row2_col1': '70b2bfdbc8630ca9ececdf74f13e3af8',
        'AI2D_TEST_sample_grid_1x2_row1_col1': '02308d4dfac6fa282ca658bff61c314f',
        'AI2D_TEST_sample_grid_1x2_row1_col2': '2a4c3674d01335f65cf6cdf2b80f9daa',
        'AI2D_TEST_sample_grid_2x2_row1_col1': 'e260e8ca73050251713e7fc406594dde',
        'AI2D_TEST_sample_grid_2x2_row1_col2': '598974cc97cafc78b95506d51a21344a',
        'AI2D_TEST_sample_grid_2x2_row2_col1': 'e06f504f46647a776fabf714ec5e58e8',
        'AI2D_TEST_sample_grid_2x2_row2_col2': '70424288b24b0ab65d03418b6feb92f8',
        'AI2D_TEST_sample_grid_3x3_row1_col1': '7d11fb01b5e398c3a5c21cec60f19769',
        'AI2D_TEST_sample_grid_3x3_row1_col2': '8046e4dcafa5c82a27ab958f4c71c76b',
        'AI2D_TEST_sample_grid_3x3_row1_col3': 'cf6983cebf55e4478092fd7bc0a62bf0',
        'AI2D_TEST_sample_grid_3x3_row2_col1': '74e3da84763458862eac8cf80019d654',
        'AI2D_TEST_sample_grid_3x3_row2_col2': '1b8f590137b10727cddc298192c72bce',
        'AI2D_TEST_sample_grid_3x3_row2_col3': 'bf62fb9c7022e6211cb86e1a2a1a4b55',
        'AI2D_TEST_sample_grid_3x3_row3_col1': '2bbe0aaddddfc909156a5c3ecffcf680',
        'AI2D_TEST_sample_grid_3x3_row3_col2': '7d16033112d05202b2beeff492b30eae',
        'AI2D_TEST_sample_grid_3x3_row3_col3': 'b4d69dcdf29372fc5c5a1f3daebf16ae',
        'AI2D_TEST_NO_MASK': 'fd8f463634d4fe9fbd23b876e8eea5be',
        # MMStar
        'MMStar': 'e1ecd2140806c1b1bbf54b43372efb9e',
        'MMStar_sample': 'e9abc21f53c9f2c2617eefed62e119e9',
        'MMStar_sample_grid_2x1_row1_col1': '2e5840e17fd5a783685ab81920df5417',
        'MMStar_sample_grid_2x1_row2_col1': '752b26ee3908f9bda9f5ce24b69aa77c',
        'MMStar_sample_grid_1x2_row1_col1': '940540e11cd565f989b792e3e4d6f9c4',
        'MMStar_sample_grid_1x2_row1_col2': '05808612e536632325e142a65c25eb13',
        'MMStar_sample_grid_2x2_row1_col1': '557dce167c259e7d579ce277c4270f04',
        'MMStar_sample_grid_2x2_row1_col2': '0f495f3b3ca81ad8facce1a73cd948d8',
        'MMStar_sample_grid_2x2_row2_col1': '32778805db3f5cd4cb085e5512b03b5d',
        'MMStar_sample_grid_2x2_row2_col2': '8ec34bf898dfc84fd9414291ce02afa3',
        'MMStar_sample_grid_3x3_row1_col1': '6fba5ff7f71df786fed6a0db1a6b9242',
        'MMStar_sample_grid_3x3_row1_col2': '930dc06e7df92492030d9178f22d1ec4',
        'MMStar_sample_grid_3x3_row1_col3': 'e633bc3c48d5f7b7bd4641954dcf8b6e',
        'MMStar_sample_grid_3x3_row2_col1': '6a46c120df4df8a7adb50fabeb4e2815',
        'MMStar_sample_grid_3x3_row2_col2': '6ec5d59ead8c94160e3d617603586601',
        'MMStar_sample_grid_3x3_row2_col3': '74352742d80f607309a601f62f66f922',
        'MMStar_sample_grid_3x3_row3_col1': '4537019c90c857b604ad399fc4177a04',
        'MMStar_sample_grid_3x3_row3_col2': '442b7f31cb5cfb4938b57544de7bd2b5',
        'MMStar_sample_grid_3x3_row3_col3': '45e9fd89fb5ea72ab39993f086c41fe4',
        # RealWorldQA
        'RealWorldQA': '4de008f55dc4fd008ca9e15321dc44b7',
        'RealWorldQA_grid_2x1_row1_col1': '23dd6bb085d7fc34b4501450285c2b65',
        'RealWorldQA_grid_2x1_row2_col1': '1bb1973c44a95f26281afcbe5c1b0344',
        'RealWorldQA_grid_1x2_row1_col1': '05f27eaaabbfda0b64510b5a40fd7904',
        'RealWorldQA_grid_1x2_row1_col2': 'e64007802642df4f3bc8bf82f3470e09',
        'RealWorldQA_grid_2x2_row1_col1': 'b9d02c2e82afbae3d8574fbebed3e1e6',
        'RealWorldQA_grid_2x2_row1_col2': '7cdc6b709903e7be066dbc7862bb9a00',
        'RealWorldQA_grid_2x2_row2_col1': '28d19d380f1e13e5266008374506c3ba',
        'RealWorldQA_grid_2x2_row2_col2': '1fe65200b3cb57d521829472f6375164',
        'RealWorldQA_grid_3x3_row1_col1': '0ef5a1f5a2caefeb623177fa5cc3dd51',
        'RealWorldQA_grid_3x3_row1_col2': '469bde7d612953e032cf46394de307dd',
        'RealWorldQA_grid_3x3_row1_col3': '73dd3dbbd3ebf85369360902acbfafbf',
        'RealWorldQA_grid_3x3_row2_col1': '8116d7ab702a9bc1307af26d6b6aa5e2',
        'RealWorldQA_grid_3x3_row2_col2': '06a9ddb5d5b6769849cb7f913e724ad8',
        'RealWorldQA_grid_3x3_row2_col3': 'a851436bcb159683591fff6c758bdb78',
        'RealWorldQA_grid_3x3_row3_col1': '0c4e03b4c7a4b9375c0811b897bfa769',
        'RealWorldQA_grid_3x3_row3_col2': '4cf68f9ed12af00e55c917f7140f0b4b',
        'RealWorldQA_grid_3x3_row3_col3': '77483b061dd0dca5d08e8cddef7f15a3',
        # BLINK
        'BLINK_sample': '47344a4acafb750141cc81e8f54a3336',
        'BLINK_sample_grid_2x1_row1_col1': '1b4e770a2c4da4eb159f598ec652ed25',
        'BLINK_sample_grid_2x1_row2_col1': '932a3202381410d29190450fcce318e7',
        'BLINK_sample_grid_1x2_row1_col1': '09b50382589fdc11498fae929877be3a',
        'BLINK_sample_grid_1x2_row1_col2': '8eb87eb260cc74628447b4b41dfebbc9',
        'BLINK_sample_grid_2x2_row1_col1': '583651bda42dafddc167b31351f47a61',
        'BLINK_sample_grid_2x2_row1_col2': '24d14bc8bc984ec2cbbffcae5a3dcdc2',
        'BLINK_sample_grid_2x2_row2_col1': '67089b263a655473fe0c96412f58b7b7',
        'BLINK_sample_grid_2x2_row2_col2': '4f5512d13c3961bdaedd53063f6f637c',
        'BLINK_sample_grid_3x3_row1_col1': '0b73032c16763a6ff9ea49dbc795ebaf',
        'BLINK_sample_grid_3x3_row1_col2': 'ab40dfec6157ecadfd0d5cca85986598',
        'BLINK_sample_grid_3x3_row1_col3': '8b05e582cf06120f5f3efb4034ff919d',
        'BLINK_sample_grid_3x3_row2_col1': '74c04da0bd4d24544db21fa719a20eba',
        'BLINK_sample_grid_3x3_row2_col2': '3cc217493df4c51a11adbabf0450420a',
        'BLINK_sample_grid_3x3_row2_col3': '0e1090451d2416fff986f226a998102f',
        'BLINK_sample_grid_3x3_row3_col1': '90ba9c3bc2219dc8bc11a80d395c444a',
        'BLINK_sample_grid_3x3_row3_col2': '5b6b16e2c14d1b4bbde328b108ea6480',
        'BLINK_sample_grid_3x3_row3_col3': '7799c7db91298c4d6911a8117b1d3efa',
        'MLLMGuard_DS': '975fc0dd7119386e198c37d71e274b3f',
        'BLINK': '3b6649b6a662184ea046908e5506260e',
        'TaskMeAnything_v1_imageqa_random': '023fef69e2ca21827afb77c5ec3bc889',
        'WorldMedQA-V': '441e63875e30c87f5750528b57b41285',
        "VisOnlyQA-VLMEvalKit": 'cf460a31d2acb8d3a7cecd0e69298bfa',
    }

    DATASET_URL.update(MMMB_URLS)
    DATASET_URL.update(MTL_MMBench_URLS)
    DATASET_MD5.update(MMMB_MD5)
    DATASET_MD5.update(MTL_MMBench_MD5)

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, report_acc_MMT, mcq_circular_eval, mcq_vanilla_eval
        # assert dataset is not None
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]
        nproc = judge_kwargs.pop('nproc', 4)

        circular = False
        if listinstr(['mmbench', 'ccbench'], dataset.lower()):
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            dump(data, eval_file)
            circular = True

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        # May have different report acc functions for different datasets
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        else:
            acc = report_acc(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        if dataset == 'AesBench_VAL':
            warnings.warn('Note that AesBench VAL is just a toy version of AesBench TEST. For full results, \
                           please evaluate on AesBench TEST. The AesBench TEST dataset is more than 20 times \
                           larger than the VAL dataset and the leaderboard results are based on AesBench TEST.')
        if dataset == 'VisOnlyQA-VLMEvalKit':
            warnings.warn('Note that the results on VisOnlyQA-VLMEvalKit are different from the results on \
                           the original VisOnlyQA. VisOnlyQA-VLMEvalKit does not include the \
                           chemistry__shape_multi split and uses a different evaluation prompt. Please \
                           explicitly specify the version of the dataset when you report results.')

        return acc


class MMMUDataset(ImageMCQDataset):

    DATASET_URL = {
        'MMMU_DEV_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv',
        'MMMU_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv',
        'MMMU_TEST_sample' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample.tsv'
        ),
        'MMMU_TEST_sample_grid_2x1_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_2x1_row1_col1.tsv'
        ),
        'MMMU_TEST_sample_grid_2x1_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_2x1_row2_col1.tsv'
        ),
        'MMMU_TEST_sample_grid_1x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_1x2_row1_col1.tsv'
        ),
        'MMMU_TEST_sample_grid_1x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_1x2_row1_col2.tsv'
        ),
        'MMMU_TEST_sample_grid_2x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_2x2_row1_col1.tsv'
        ),
        'MMMU_TEST_sample_grid_2x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_2x2_row1_col2.tsv'
        ),
        'MMMU_TEST_sample_grid_2x2_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_2x2_row2_col1.tsv'
        ),
        'MMMU_TEST_sample_grid_2x2_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_2x2_row2_col2.tsv'
        ),
        'MMMU_TEST_sample_grid_3x3_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_3x3_row1_col1.tsv'
        ),
        'MMMU_TEST_sample_grid_3x3_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_3x3_row1_col2.tsv'
        ),
        'MMMU_TEST_sample_grid_3x3_row1_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_3x3_row1_col3.tsv'
        ),
        'MMMU_TEST_sample_grid_3x3_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_3x3_row2_col1.tsv'
        ),
        'MMMU_TEST_sample_grid_3x3_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_3x3_row2_col2.tsv'
        ),
        'MMMU_TEST_sample_grid_3x3_row2_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_3x3_row2_col3.tsv'
        ),
        'MMMU_TEST_sample_grid_3x3_row3_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_3x3_row3_col1.tsv'
        ),
        'MMMU_TEST_sample_grid_3x3_row3_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_3x3_row3_col2.tsv'
        ),
        'MMMU_TEST_sample_grid_3x3_row3_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MMMU_TEST/MMMU_TEST_sample_grid_3x3_row3_col3.tsv'
        ),
    }

    DATASET_MD5 = {
        'MMMU_DEV_VAL': '585e8ad75e73f75dcad265dfd0417d64',
        'MMMU_TEST': 'c19875d11a2d348d07e5eb4bdf33166d',
        'MMMU_TEST_sample': 'ec6eaf53d76d0f47efa85b14bb50ed1a',
        'MMMU_TEST_sample_grid_2x1_row1_col1': '2340fa1a2c4a7ca10d8f85f7243a7033',
        'MMMU_TEST_sample_grid_2x1_row2_col1': '44d695e397f2f638a47721b95c450145',
        'MMMU_TEST_sample_grid_1x2_row1_col1': 'a29ff0d59ff353f54ee505f28138a450',
        'MMMU_TEST_sample_grid_1x2_row1_col2': 'bfd9084b47d647783b58466ee115de5f',
        'MMMU_TEST_sample_grid_2x2_row1_col1': '71c0090df63864c514d38c6236b64b22',
        'MMMU_TEST_sample_grid_2x2_row1_col2': '9b3a08175e661a04a7733e2c928fd21e',
        'MMMU_TEST_sample_grid_2x2_row2_col1': '9a7e622704c77264d12b61fe9d57b809',
        'MMMU_TEST_sample_grid_2x2_row2_col2': '69581ec0eb8dcde2e79e00a3517a26d9',
        'MMMU_TEST_sample_grid_3x3_row1_col1': '99cb51fdb9bf0c862b2f6155392e44f3',
        'MMMU_TEST_sample_grid_3x3_row1_col2': '5ad27adb5046bed93285b6895ca81059',
        'MMMU_TEST_sample_grid_3x3_row1_col3': 'b286654f886ce1d95f23f07b03047e5a',
        'MMMU_TEST_sample_grid_3x3_row2_col1': '5dbfa71383dbd045b3ad96450fc7fb62',
        'MMMU_TEST_sample_grid_3x3_row2_col2': 'ee9f530e8403ad88d1474c09f19874af',
        'MMMU_TEST_sample_grid_3x3_row2_col3': '02d920af2e360dd1cb9307b14b4acad0',
        'MMMU_TEST_sample_grid_3x3_row3_col1': '682cbfafdb491bf0a5e87e9fa3832a3e',
        'MMMU_TEST_sample_grid_3x3_row3_col2': 'c2cbc47e9442d42cf508290ca641f031',
        'MMMU_TEST_sample_grid_3x3_row3_col3': '7380887ef1cf06c482d0b871bedf54d1',
    }

    @staticmethod
    def split_MMMU(msgs):
        text, images = None, []
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None
                text = s['value']
        text_segs = text.split('<image ')
        if len(text_segs) == 1:
            return msgs

        segs = [dict(type='text', value=text_segs[0])]
        for i, seg in enumerate(text_segs):
            if i == 0:
                continue
            assert istype(seg[0], int) and seg[1] == '>'
            image_idx = int(seg[0]) - 1
            segs.append(dict(type='image', value=images[image_idx]))
            segs.append(dict(type='text', value=seg[2:]))
        return segs

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        msgs = self.split_MMMU(msgs)
        return msgs


class MUIRDataset(ImageMCQDataset):

    DATASET_URL = {
        'MUIRBench': 'http://opencompass.openxxlab.com/utils/VLMEval/MUIRBench.tsv',
        'MUIRBench_sample' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample.tsv'
        ),
        'MUIRBench_sample_grid_2x1_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_2x1_row1_col1.tsv'
        ),
        'MUIRBench_sample_grid_2x1_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_2x1_row2_col1.tsv'
        ),
        'MUIRBench_sample_grid_1x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_1x2_row1_col1.tsv'
        ),
        'MUIRBench_sample_grid_1x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_1x2_row1_col2.tsv'
        ),
        'MUIRBench_sample_grid_2x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_2x2_row1_col1.tsv'
        ),
        'MUIRBench_sample_grid_2x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_2x2_row1_col2.tsv'
        ),
        'MUIRBench_sample_grid_2x2_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_2x2_row2_col1.tsv'
        ),
        'MUIRBench_sample_grid_2x2_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_2x2_row2_col2.tsv'
        ),
        'MUIRBench_sample_grid_3x3_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_3x3_row1_col1.tsv'
        ),
        'MUIRBench_sample_grid_3x3_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_3x3_row1_col2.tsv'
        ),
        'MUIRBench_sample_grid_3x3_row1_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_3x3_row1_col3.tsv'
        ),
        'MUIRBench_sample_grid_3x3_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_3x3_row2_col1.tsv'
        ),
        'MUIRBench_sample_grid_3x3_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_3x3_row2_col2.tsv'
        ),
        'MUIRBench_sample_grid_3x3_row2_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_3x3_row2_col3.tsv'
        ),
        'MUIRBench_sample_grid_3x3_row3_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_3x3_row3_col1.tsv'
        ),
        'MUIRBench_sample_grid_3x3_row3_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_3x3_row3_col2.tsv'
        ),
        'MUIRBench_sample_grid_3x3_row3_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MUIRBench/MUIRBench_sample_grid_3x3_row3_col3.tsv'
        ), 
        
    }

    DATASET_MD5 = {
        'MUIRBench': '2e5e6fd7699761b08a7cb3ab8c0c2ec8',
        'MUIRBench_sample': '4a5fa3b204f09c05d5de101948ae54ae',
        'MUIRBench_sample_grid_2x1_row1_col1': 'b83b1c4c318e6c8fd831c9ca4951c4fb',
        'MUIRBench_sample_grid_2x1_row2_col1': 'bb11a02a4aefc70d3915164b33efd27b',
        'MUIRBench_sample_grid_1x2_row1_col1': '066b03710781809d609a6cf576c643db',
        'MUIRBench_sample_grid_1x2_row1_col2': 'b5b7007415f57ef7354da5b3bb67cc4a',
        'MUIRBench_sample_grid_2x2_row1_col1': 'a46850b6a9bedd4b19dbda7eaa8a298f',
        'MUIRBench_sample_grid_2x2_row1_col2': '29a9677266dfd15877370a7fa502b2b3',
        'MUIRBench_sample_grid_2x2_row2_col1': '9193a5b94f4a20274b0cb057cccfe9ae',
        'MUIRBench_sample_grid_2x2_row2_col2': '387e94875dbf157672594cbbd5bcd07c',
        'MUIRBench_sample_grid_3x3_row1_col1': '80e48880efcbf33d18dd711df6d3cfe0',
        'MUIRBench_sample_grid_3x3_row1_col2': '6cb99b19763d48379296c2a12ae498c3',
        'MUIRBench_sample_grid_3x3_row1_col3': 'be3a4b962134f796e709860149d8709c',
        'MUIRBench_sample_grid_3x3_row2_col1': 'f8da5cee44b8faee2b1860dec24efa0a',
        'MUIRBench_sample_grid_3x3_row2_col2': '2ddfd0eeece9f4fdd2a896cce3a50fa0',
        'MUIRBench_sample_grid_3x3_row2_col3': '6977690e170b46facf6b188fcb3a4c7e',
        'MUIRBench_sample_grid_3x3_row3_col1': 'd5b9a37011b81c9943576a8063782745',
        'MUIRBench_sample_grid_3x3_row3_col2': 'bcddcd03bb4d537b98c2b59ab284158b',
        'MUIRBench_sample_grid_3x3_row3_col3': 'a94080cacd5af5ce4a72243bed1a93b1',
        
    }

    @staticmethod
    def split_MUIR(msgs):
        text, images = None, []

        # Separate images and text from msgs
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None  # Ensure only one text entry is expected
                text = s['value']

        # Split text by <image> tags
        text_segs = text.split('<image>')

        # Initialize the segments list
        segs = []

        # Iterate through the text segments and images
        for i, seg in enumerate(text_segs):
            # Append the image if this is not the first segment and there are still images left
            if i > 0 and i - 1 < len(images):
                segs.append(dict(type='image', value=images[i - 1]))
            # Append the text segment (if it's non-empty)
            if len(seg) > 0:
                segs.append(dict(type='text', value=seg))

        return segs

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        # options_prompt = ''
        options_prompt = '\n'.join([f'{key}. {item}' for key, item in options.items()])
        # for key, item in options.items():
        #     options_prompt += f'{key}. {item}\n'

        prompt = ''

        prompt += f'{question}\n'
        if len(options):
            prompt += options_prompt
            prompt += "\nAnswer with the option's letter from the given choices directly."

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        msgs = self.split_MUIR(msgs)
        return msgs


class GMAIMMBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'GMAI-MMBench_VAL': 'https://huggingface.co/datasets/VLMEval/GMAI-MMBench/resolve/main/GMAI-MMBench_VAL.tsv',
        'GMAI_mm_bench_TEST_part_1': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_1.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_2': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_2.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_3': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_3.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_4': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_4.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_5': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_5.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_6': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_6.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_7': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_7.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_8': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_8.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_9': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_9.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_10': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_10.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_11': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_11.tsv',  # noqa: E501
        'GMAI-MMBench_VAL_sample' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_2x1_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_2x1_row1_col1.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_2x1_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_2x1_row2_col1.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_1x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_1x2_row1_col1.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_1x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_1x2_row1_col2.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_2x2_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_2x2_row1_col1.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_2x2_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_2x2_row1_col2.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_2x2_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_2x2_row2_col1.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_2x2_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_2x2_row2_col2.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_3x3_row1_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_3x3_row1_col1.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_3x3_row1_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_3x3_row1_col2.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_3x3_row1_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_3x3_row1_col3.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_3x3_row2_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_3x3_row2_col1.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_3x3_row2_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_3x3_row2_col2.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_3x3_row2_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_3x3_row2_col3.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_3x3_row3_col1' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_3x3_row3_col1.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_3x3_row3_col2' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_3x3_row3_col2.tsv'
        ),
        'GMAI-MMBench_VAL_sample_grid_3x3_row3_col3' : (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/GMAI-MMBench_VAL/GMAI-MMBench_VAL_sample_grid_3x3_row3_col3.tsv'
        ), 
    }

    DATASET_MD5 = {
        'GMAI-MMBench_VAL': '254bd581627866f1c499d3d6b4422324',
        'GMAI_mm_bench_TEST_part_1': '900d735231230a63f4ed45665c078ef4',
        'GMAI_mm_bench_TEST_part_2': '1b27ab621386945d7e4a765ad2d22b0e',
        'GMAI_mm_bench_TEST_part_3': '44bdc2b6267dd505d529b8cad06f0fb2',
        'GMAI_mm_bench_TEST_part_4': '5a04a04fcac9f1466709f242fdb80acb',
        'GMAI_mm_bench_TEST_part_5': 'c70baf8909eda9af0ddeab275c721336',
        'GMAI_mm_bench_TEST_part_6': '825abc39596b644dead9350d0cfa3b96',
        'GMAI_mm_bench_TEST_part_7': 'defb8aed2fb77365a76b6b9abd6a2701',
        'GMAI_mm_bench_TEST_part_8': 'ff490d60b85f2bb0abb67a435b298c65',
        'GMAI_mm_bench_TEST_part_9': 'ff67c86f40da93b09139ac1d1ba5dc6b',
        'GMAI_mm_bench_TEST_part_10': '3dae94627b9ac0fe00180d4780fbf6dc',
        'GMAI_mm_bench_TEST_part_11': 'd08dc813f0eb6bbab63cae2a9d113c4b',
        'GMAI-MMBench_VAL_sample': '',
        'GMAI-MMBench_VAL_sample_grid_2x1_row1_col1': '',
        'GMAI-MMBench_VAL_sample_grid_2x1_row2_col1': '',
        'GMAI-MMBench_VAL_sample_grid_1x2_row1_col1': '',
        'GMAI-MMBench_VAL_sample_grid_1x2_row1_col2': '',
        'GMAI-MMBench_VAL_sample_grid_2x2_row1_col1': '',
        'GMAI-MMBench_VAL_sample_grid_2x2_row1_col2': '',
        'GMAI-MMBench_VAL_sample_grid_2x2_row2_col1': '',
        'GMAI-MMBench_VAL_sample_grid_2x2_row2_col2': '',
        'GMAI-MMBench_VAL_sample_grid_3x3_row1_col1': '',
        'GMAI-MMBench_VAL_sample_grid_3x3_row1_col2': '',
        'GMAI-MMBench_VAL_sample_grid_3x3_row1_col3': '',
        'GMAI-MMBench_VAL_sample_grid_3x3_row2_col1': '',
        'GMAI-MMBench_VAL_sample_grid_3x3_row2_col2': '',
        'GMAI-MMBench_VAL_sample_grid_3x3_row2_col3': '',
        'GMAI-MMBench_VAL_sample_grid_3x3_row3_col1': '',
        'GMAI-MMBench_VAL_sample_grid_3x3_row3_col2': '',
        'GMAI-MMBench_VAL_sample_grid_3x3_row3_col3': '',
    }

    @classmethod
    def supported_datasets(cls):
        return ['GMAI-MMBench_VAL', 'GMAI-MMBench_TEST']

    def load_data(self, dataset):
        if dataset == 'GMAI-MMBench_VAL':
            data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
            if file_size(data_path, 'GB') > 1:
                local_path = data_path.replace('.tsv', '_local.tsv')
                if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                    from ..tools import LOCALIZE
                    LOCALIZE(data_path, local_path)
                data_path = local_path
            return load(data_path)
        elif dataset == 'GMAI-MMBench_TEST':
            dfs = []
            for part_num in range(1, 12):
                part_name = f'GMAI_mm_bench_TEST_part_{part_num}'
                url = self.DATASET_URL[part_name]
                file_md5 = self.DATASET_MD5.get(part_name)
                tsv_path = osp.join(LMUDataRoot(), f'{part_name}.tsv')
                if not osp.exists(tsv_path) or (file_md5 and md5(tsv_path) != file_md5):
                    download_file(url, filename=tsv_path)
                local_path = tsv_path.replace('.tsv', '_local.tsv')
                if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                    from ..tools import LOCALIZE
                    LOCALIZE(tsv_path, local_path)
                tsv_path = local_path
                # 加载数据
                df = load(tsv_path)
                dfs.append(df)
            # 合并所有数据
            data = pd.concat(dfs, ignore_index=True)
            return data
        else:
            raise ValueError(f"未知的数据集：{dataset}")

    def report_acc_by_groups(self, df, group_column):
        res = defaultdict(list)

        # Check for the 'split' column
        if 'split' in df:
            splits = list(set(df['split']))
            res['split'] = splits
        else:
            df['split'] = ['none'] * len(df)
            res['split'] = ['none']

        res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]

        if group_column not in df:
            raise ValueError(f"Column '{group_column}' not found in dataframe.")  # noqa: E713

        abilities = list(set(df[group_column]))
        abilities = ['None' if isinstance(ab, float) and pd.isna(ab) else ab for ab in abilities]
        abilities.sort()

        for ab in abilities:
            ab_name = ab
            sub_df = df[df[group_column] == ab]
            res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

        return pd.DataFrame(res)

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, mcq_vanilla_eval
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        acc = report_acc(data)

        for group_col in ['clinical vqa task', 'department', 'perceptual granularity']:
            acc_grouped = self.report_acc_by_groups(data, group_col)
            score_file_grouped = eval_file.replace(f'.{suffix}', f'_{group_col}_acc.csv')
            dump(acc_grouped, score_file_grouped)

        return acc


class MMERealWorld(ImageMCQDataset):

    TYPE = 'MMERealWorld'

    DATASET_MD5 = {
        'MME-RealWorld': '271c33ec814c39533c467ec6fb8a6f36',
        'MME-RealWorld-Lite': '4c17057d7d3b6c4a0d4397c3dae0881c',
        'MME-RealWorld-CN': 'daaa763d52a760a38606d5dedb3fe444',
    }
    SYS = {
        'MME-RealWorld': (
            'Select the best answer to the above multiple-choice question based on the image. '
            'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
            'The best answer is:'
        ),
        'MME-RealWorld-Lite': (
            'Select the best answer to the above multiple-choice question based on the image. '
            'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
            'The best answer is:'
        ),
        'MME-RealWorld-CN': (
            '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n'
            '最佳答案为：'
        ),
    }

    @classmethod
    def supported_datasets(cls):
        return ['MME-RealWorld', 'MME-RealWorld-CN', 'MME-RealWorld-Lite',]

    def load_data(
        self, dataset="MME-RealWorld", repo_id="yifanzhang114/MME-RealWorld-Base64"
    ):

        def check_integrity(pth):
            data_file = osp.join(pth, f"{dataset}.tsv")

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.DATASET_MD5[dataset]:
                return False
            return True

        def generate_tsv(pth):
            tsv_file = os.path.join(pth, f"{dataset}.tsv")

            if os.path.exists(tsv_file):
                print(f"{tsv_file} already exists.")
                return

            json_dir = os.path.join(pth, dataset)
            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

            data_list = []
            for json_file in json_files:
                with open(os.path.join(json_dir, json_file), "r") as f:
                    data = json.load(f)
                    for item in tqdm(data):
                        choice_prompt = (
                            "The choices are listed below:\n"
                            if dataset in ["MME-RealWorld", "MME-RealWorld-Lite"]
                            else "选项如下所示:\n"
                        )
                        data_list.append(
                            {
                                "index": item["index"],
                                "image": item["image"],
                                "question": item["question"],
                                "multi-choice options": choice_prompt
                                + "\n".join(item["multi-choice options"]),
                                "A": item["multi-choice options"][0][4:],
                                "B": item["multi-choice options"][1][4:],
                                "C": item["multi-choice options"][2][4:],
                                "D": item["multi-choice options"][3][4:],
                                "E": item["multi-choice options"][4][4:],
                                "answer": item["answer"],
                                "category": item["category"],
                                "l2-category": item["l2-category"],
                            }
                        )
            df = pd.DataFrame(data_list)
            df.to_csv(tsv_file, sep="\t", index=False)
            print(f"TSV file saved to {tsv_file}")

        # Check if dataset is cached and has integrity
        if dataset == "MME-RealWorld-Lite":
            url = 'https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Base64/resolve/main/mme_realworld_lite.tsv'  # noqa: E501
            file_md5 = (
                self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
            )
            datas = self.prepare_tsv(url, file_md5)
            choice_prompt = "The choices are listed below:\n"
            for index, item in datas.iterrows():
                options = eval(item["multi-choice options"])
                datas.loc[index, "multi-choice options"] = choice_prompt + "\n".join(
                    options
                )
                datas.loc[index, "A"] = options[0][4:]
                datas.loc[index, "B"] = options[1][4:]
                datas.loc[index, "C"] = options[2][4:]
                datas.loc[index, "D"] = options[3][4:]
                datas.loc[index, "E"] = options[4][4:]
            return datas

        update_flag = False
        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
            print(f"Using cached dataset from {cache_path}")
        else:
            from huggingface_hub import snapshot_download

            # Download or find the dataset path
            dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
            generate_tsv(dataset_path)
            update_flag = True

        data_path = os.path.join(dataset_path, f"{dataset}.tsv")
        if file_size(data_path, "GB") > 1:
            local_path = data_path.replace(".tsv", "_local.tsv")
            if (
                not osp.exists(local_path)
                or os.environ.get("FORCE_LOCAL", None)
                or update_flag
            ):
                from vlmeval.tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def post_build(self, dataset):
        self.TYPE = 'MMERealWorld'

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        choice_prompt = line['multi-choice options'] + '\n'
        question += ' ' + choice_prompt + self.SYS[self.dataset_name]

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import extract_characters_regex, get_dimension_rating
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        FAIL_MSG = 'Failed to obtain answer via API.'
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):

            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            cnt_rejected = 0
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]

                extract_pred = extract_characters_regex(pred)
                if extract_pred == '':
                    cnt_rejected += 1
                    data.loc[data['index'] == idx, 'score'] = 0
                else:
                    data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {cnt_rejected} questions. '
                f'Those questions will be counted as 0 score in ALL rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating


class HRBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'HRBench4K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.tsv',
        'HRBench8K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.tsv',
    }

    DATASET_MD5 = {
        'HRBench4K': 'f6b041b03d49543494b8a56d2e35be65',
        'HRBench8K': '274c9c7f89329b804a4723178a00219c',
    }

    def evaluate(self, eval_file, **judge_kwargs):
        assert os.path.exists(eval_file), '{} does not exist!'.format(eval_file)
        from .utils.multiple_choice import mcq_vanilla_eval
        from .utils.hrbench import report_acc_hrbench
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'extract_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')

        if osp.exists(score_file):
            acc = load(score_file)
            return acc
        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        acc = report_acc_hrbench(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc


class CustomMCQDataset(ImageMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)


class NaturalBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'NaturalBenchDataset': (
            'https://huggingface.co/datasets/BaiqiL/'
            'NaturalBench/resolve/main/NaturalBenchDataset.tsv'
        ),
    }
    DATASET_MD5 = {
        'NaturalBenchDataset':'dbe25b044bc35696426381e9ba4fe930',
    }

    def build_prompt(self, line):
        SUFFIX_FOR_VQA = {
            "yes_no": "Please answer Yes or No.",
            "multiple_choice": "Please output the letter corresponding to the correct option."
        }
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        prompt = f'{question} {SUFFIX_FOR_VQA[line["type"]]}'
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.naturalbench import extract_answer, get_scores

        data = load(eval_file)
        data = data.sort_values(by='index')
        predictions = [str(x) for x in data['prediction']]
        answers = [str(x) for x in data['answer']]
        indexs = [str(x) for x in data['index']]
        meta = self.data
        types = [str(x) for x in meta['type']]
        results = {}
        assert len(predictions) == len(answers) == len(indexs) == len(types) == (1900 * 4)
        number_answered_samples = len(predictions) // 4
        for i in range(number_answered_samples):
            results[i] = {
                "q0_i0": extract_answer(predictions[i * 4], types[i * 4]),
                "q0_i1": extract_answer(predictions[i * 4 + 1], types[i * 4 + 1]),
                "q1_i0": extract_answer(predictions[i * 4 + 2], types[i * 4 + 2]),
                "q1_i1": extract_answer(predictions[i * 4 + 3], types[i * 4 + 3])
            }

        scores = get_scores(results)
        print(scores)
        score_file = 'NaturalBench_acc.csv'
        df = pd.DataFrame(list(scores.items()), columns=['Metric', 'Score'])
        dump(df, score_file)

        return scores
