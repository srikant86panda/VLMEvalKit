from ..smp import *
from ..utils import *
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE


class ImageYORNDataset(ImageBaseDataset):

    TYPE = 'Y/N'

    DATASET_URL = {
        'MME': 'https://opencompass.openxlab.space/utils/VLMEval/MME.tsv',
        'MME_sample': 'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/resolve/main/MME/MME_sample.tsv',
        'MME_sample_grid_1x2_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_1x2_row1_col1.tsv'
        ),
        'MME_sample_grid_1x2_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_1x2_row1_col2.tsv'
        ),
        'MME_sample_grid_2x1_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_2x1_row1_col1.tsv'
        ),
        'MME_sample_grid_2x1_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_2x1_row2_col1.tsv'
        ),
        'MME_sample_grid_2x2_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_2x2_row1_col1.tsv'                                 
        ),
        'MME_sample_grid_2x2_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_2x2_row1_col2.tsv'
        ),
        'MME_sample_grid_2x2_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_2x2_row2_col1.tsv'
        ),
        'MME_sample_grid_2x2_row2_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_2x2_row2_col2.tsv'
        ),
        'MME_sample_grid_3x3_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_3x3_row1_col1.tsv'
        ),
        'MME_sample_grid_3x3_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_3x3_row1_col2.tsv'
        ),
        'MME_sample_grid_3x3_row1_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_3x3_row1_col3.tsv'
        ),
        'MME_sample_grid_3x3_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_3x3_row2_col1.tsv'
        ),
        'MME_sample_grid_3x3_row2_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_3x3_row2_col2.tsv'
        ),
        'MME_sample_grid_3x3_row2_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_3x3_row2_col3.tsv'
        ),
        'MME_sample_grid_3x3_row3_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_3x3_row3_col1.tsv'
        ),
        'MME_sample_grid_3x3_row3_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_3x3_row3_col2.tsv')
        ,
        'MME_sample_grid_3x3_row3_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/MME/MME_sample_grid_3x3_row3_col3.tsv'
        ),
        'HallusionBench': 'https://opencompass.openxlab.space/utils/VLMEval/HallusionBench.tsv',
        'HallusionBench_sample': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample.tsv'
        ),
        'HallusionBench_sample_grid_1x2_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_1x2_row1_col1.tsv'
        ),
        'HallusionBench_sample_grid_1x2_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_1x2_row1_col2.tsv'
        ),
        'HallusionBench_sample_grid_2x1_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_2x1_row1_col1.tsv'
        ),
        'HallusionBench_sample_grid_2x1_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_2x1_row2_col1.tsv'
        ),
        'HallusionBench_sample_grid_2x2_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_2x2_row1_col1.tsv'                                 
        ),
        'HallusionBench_sample_grid_2x2_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_2x2_row1_col2.tsv'
        ),
        'HallusionBench_sample_grid_2x2_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_2x2_row2_col1.tsv'
        ),
        'HallusionBench_sample_grid_2x2_row2_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_2x2_row2_col2.tsv'
        ),
        'HallusionBench_sample_grid_3x3_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_3x3_row1_col1.tsv'
        ),
        'HallusionBench_sample_grid_3x3_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_3x3_row1_col2.tsv'
        ),
        'HallusionBench_sample_grid_3x3_row1_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_3x3_row1_col3.tsv'
        ),
        'HallusionBench_sample_grid_3x3_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_3x3_row2_col1.tsv'
        ),
        'HallusionBench_sample_grid_3x3_row2_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_3x3_row2_col2.tsv'
        ),
        'HallusionBench_sample_grid_3x3_row2_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_3x3_row2_col3.tsv'
        ),
        'HallusionBench_sample_grid_3x3_row3_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_3x3_row3_col1.tsv'
        ),
        'HallusionBench_sample_grid_3x3_row3_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_3x3_row3_col2.tsv')
        ,
        'HallusionBench_sample_grid_3x3_row3_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/HallusionBench/HallusionBench_sample_grid_3x3_row3_col3.tsv'
        ),
        'POPE': 'https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv',
        'POPE_sample': 'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/resolve/main/POPE/POPE_sample.tsv',
        'POPE_sample_grid_1x2_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_1x2_row1_col1.tsv'
        ),
        'POPE_sample_grid_1x2_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_1x2_row1_col2.tsv'
        ),
        'POPE_sample_grid_2x1_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_2x1_row1_col1.tsv'
        ),
        'POPE_sample_grid_2x1_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_2x1_row2_col1.tsv'
        ),
        'POPE_sample_grid_2x2_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_2x2_row1_col1.tsv'                                 
        ),
        'POPE_sample_grid_2x2_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_2x2_row1_col2.tsv'
        ),
        'POPE_sample_grid_2x2_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_2x2_row2_col1.tsv'
        ),
        'POPE_sample_grid_2x2_row2_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_2x2_row2_col2.tsv'
        ),
        'POPE_sample_grid_3x3_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_3x3_row1_col1.tsv'
        ),
        'POPE_sample_grid_3x3_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_3x3_row1_col2.tsv'
        ),
        'POPE_sample_grid_3x3_row1_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_3x3_row1_col3.tsv'
        ),
        'POPE_sample_grid_3x3_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_3x3_row2_col1.tsv'
        ),
        'POPE_sample_grid_3x3_row2_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_3x3_row2_col2.tsv'
        ),
        'POPE_sample_grid_3x3_row2_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_3x3_row2_col3.tsv'
        ),
        'POPE_sample_grid_3x3_row3_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_3x3_row3_col1.tsv'
        ),
        'POPE_sample_grid_3x3_row3_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_3x3_row3_col2.tsv')
        ,
        'POPE_sample_grid_3x3_row3_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/POPE/POPE_sample_grid_3x3_row3_col3.tsv'
        ),
        'AMBER': 'https://huggingface.co/datasets/yifanzhang114/AMBER_base64/resolve/main/AMBER.tsv',
        'AMBER_sample': 'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/resolve/main/POPE/POPE_sample.tsv',
        'AMBER_sample_grid_1x2_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_1x2_row1_col1.tsv'
        ),
        'AMBER_sample_grid_1x2_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_1x2_row1_col2.tsv'
        ),
        'AMBER_sample_grid_2x1_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_2x1_row1_col1.tsv'
        ),
        'AMBER_sample_grid_2x1_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_2x1_row2_col1.tsv'
        ),
        'AMBER_sample_grid_2x2_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_2x2_row1_col1.tsv'                                 
        ),
        'AMBER_sample_grid_2x2_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_2x2_row1_col2.tsv'
        ),
        'AMBER_sample_grid_2x2_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_2x2_row2_col1.tsv'
        ),
        'AMBER_sample_grid_2x2_row2_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_2x2_row2_col2.tsv'
        ),
        'AMBER_sample_grid_3x3_row1_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_3x3_row1_col1.tsv'
        ),
        'AMBER_sample_grid_3x3_row1_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_3x3_row1_col2.tsv'
        ),
        'AMBER_sample_grid_3x3_row1_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_3x3_row1_col3.tsv'
        ),
        'AMBER_sample_grid_3x3_row2_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_3x3_row2_col1.tsv'
        ),
        'AMBER_sample_grid_3x3_row2_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_3x3_row2_col2.tsv'
        ),
        'AMBER_sample_grid_3x3_row2_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_3x3_row2_col3.tsv'
        ),
        'AMBER_sample_grid_3x3_row3_col1': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_3x3_row3_col1.tsv'
        ),
        'AMBER_sample_grid_3x3_row3_col2': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_3x3_row3_col2.tsv')
        ,
        'AMBER_sample_grid_3x3_row3_col3': (
            'https://huggingface.co/datasets/hiteshpatel945/VLMEVAL/'
            'resolve/main/AMBER/AMBER_sample_grid_3x3_row3_col3.tsv'
        ),
    }

    DATASET_MD5 = {
        'MME': 'b36b43c3f09801f5d368627fb92187c3',
        'MME_sample': 'b36b43c3f09801f5d368627fb92187c3',
        'MME_sample_grid_1x2_row1_col1': 'b26af6e44af28d0d17eb6580bf37778d',
        'MME_sample_grid_1x2_row1_col2': '764bccd4fec149227e1cab3a734cec27',
        'MME_sample_grid_2x1_row1_col1': 'aa26beea01c2e0b0dff7a920ea17c591',
        'MME_sample_grid_2x1_row2_col1': '34a049f5ec2aa831f3a53bea14d02726',
        'MME_sample_grid_2x2_row1_col1': '8d8d4c141a964d0c1c0364e1b42af8d0',
        'MME_sample_grid_2x2_row1_col2': 'b9189329c929179e71f260f349bc011a',
        'MME_sample_grid_2x2_row2_col1': '31dac8711b1f819b3101766ba5d56a0e',
        'MME_sample_grid_2x2_row2_col2': '0f49e64dc44d1c1a43798722feaf041a',
        'MME_sample_grid_3x3_row1_col1': '03fa7af7ed0d135fd3fde8e86fe08d5e',
        'MME_sample_grid_3x3_row1_col2': 'dab7f7305d4f67532ba7b0c61f4ee01b',
        'MME_sample_grid_3x3_row1_col3': 'd306177421d08a0532862d4f06ed7f3e',
        'MME_sample_grid_3x3_row2_col1': '4928e6833f7c9cc4c91696e53246673a',
        'MME_sample_grid_3x3_row2_col2': '3dcd7cf04f5a47fb4f4f85db225ae7b2',
        'MME_sample_grid_3x3_row2_col3': '857fce019cc1a8f2dd32376db241deb9',
        'MME_sample_grid_3x3_row3_col1': '21e1f7e08fc9fa7e1826392945b88521',
        'MME_sample_grid_3x3_row3_col2': '40e3c0a71b03a7e77c8d08b58c24b05f',
        'MME_sample_grid_3x3_row3_col3': 'b1355530add797e6a5428aeeb4ca2425',
        'HallusionBench': '0c23ac0dc9ef46832d7a24504f2a0c7c',
        'HallusionBench_sample': 'f641e821f0b0561a40630eb056079b8c',
        'HallusionBench_sample_grid_1x2_row1_col1': '7dcf3e82059682643af4c7fc57717085',
        'HallusionBench_sample_grid_1x2_row1_col2': '85552654f8d39c8e06eeb8403df7ca24',
        'HallusionBench_sample_grid_2x1_row1_col1': '67707df360fa071bb6574f4169b3d96f',
        'HallusionBench_sample_grid_2x1_row2_col1': '571fe31e8ccbe4577b578fbff8036baf',
        'HallusionBench_sample_grid_2x2_row1_col1': 'b03d1ac96a3863a08ca979604571a653',
        'HallusionBench_sample_grid_2x2_row1_col2': 'd293b189dca3c14319c5b9614958b4e4',
        'HallusionBench_sample_grid_2x2_row2_col1': '7414dbec2155d21fc4b29e033444f7a9',
        'HallusionBench_sample_grid_2x2_row2_col2': '110d21629a6a92086c191ccedfd57963',
        'HallusionBench_sample_grid_3x3_row1_col1': 'c5beb9bf5ed1251cb42efc0a19389232',
        'HallusionBench_sample_grid_3x3_row1_col2': '3d06a26580e209fe7467c727ebdf7944',
        'HallusionBench_sample_grid_3x3_row1_col3': '3ea9ccf2e55f7af5db3d9b3665b91ff4',
        'HallusionBench_sample_grid_3x3_row2_col1': '949dae3a2fd96b0e3f5b2c0df3900b46',
        'HallusionBench_sample_grid_3x3_row2_col2': '9a4ae54931f5572295e677d82bb1168b',
        'HallusionBench_sample_grid_3x3_row2_col3': '21318861cfd04cd795d50f0d8b26161f',
        'HallusionBench_sample_grid_3x3_row3_col1': '54bcc49451f71299298f8fe58eadfd0e',
        'HallusionBench_sample_grid_3x3_row3_col2': '6db4b7a99d0e125a3cf4d91c8981f83a',
        'HallusionBench_sample_grid_3x3_row3_col3': '252ba054fa5d089b4458a6ff6447d680',
        'POPE': 'c12f5acb142f2ef1f85a26ba2fbe41d5',
        'POPE_sample': 'c12f5acb142f2ef1f85a26ba2fbe41d5',
        'POPE_sample_grid_1x2_row1_col1': '871f1219c5e454832585527c14441839',
        'POPE_sample_grid_1x2_row1_col2': 'e0c708f15d860d365c5d10c5ee84379c',
        'POPE_sample_grid_2x1_row1_col1': 'dbda86318b62deb904c128a193fd55eb',
        'POPE_sample_grid_2x1_row2_col1': 'fd737cee55814d201a754441304d8f42',
        'POPE_sample_grid_2x2_row1_col1': 'de499cb734d90d89bea812a889dfab8b',
        'POPE_sample_grid_2x2_row1_col2': '46da52b6430dce9927ad37978d15b364',
        'POPE_sample_grid_2x2_row2_col1': 'c02c0d26153f1e19fdad5428e94fe204',
        'POPE_sample_grid_2x2_row2_col2': 'c57332bab0e651cdfde8517bcd249c5e',
        'POPE_sample_grid_3x3_row1_col1': '833154f007ecb07dfd15fa89fe8ae1b0',
        'POPE_sample_grid_3x3_row1_col2': '9f5895b645a6f641f9d6448a744717ed',
        'POPE_sample_grid_3x3_row1_col3': '82428a95c0c23f3a90f4440bc77e7746',
        'POPE_sample_grid_3x3_row2_col1': '71a4ab23dc34ecb755933ffc678711cd',
        'POPE_sample_grid_3x3_row2_col2': 'bfe37eacbdd3c2800df94746f7eb11cf',
        'POPE_sample_grid_3x3_row2_col3': 'b50fdc2fa0abc0fa2153d3e154d612f2',
        'POPE_sample_grid_3x3_row3_col1': 'dfbb4278b4819c82ff62bdc848ad54a6',
        'POPE_sample_grid_3x3_row3_col2': '7a0c0273450c789e01949d26c5233cd1',
        'POPE_sample_grid_3x3_row3_col3': '9496b532d8d1d7e7efae64db3c97bd3a',
        'AMBER': '970d94c0410916166e0a76ba75da7934',
        'AMBER_sample': '970d94c0410916166e0a76ba75da7934',
        'AMBER_sample_grid_1x2_row1_col1': '4f2e6289c7881d03e7157e73ff6ce528',
        'AMBER_sample_grid_1x2_row1_col2': '564a0db4bc67c548146008610e670533',
        'AMBER_sample_grid_2x1_row1_col1': '9d8ffd37b410d62dd74217fcf54d25fd',
        'AMBER_sample_grid_2x1_row2_col1': '0e6209de0d4c2f169b137f735c90dfaf',
        'AMBER_sample_grid_2x2_row1_col1': 'd3254befcd6a0c6a03daed9eedae096e',
        'AMBER_sample_grid_2x2_row1_col2': '9030e586a768645669fa9cd1a6baebab',
        'AMBER_sample_grid_2x2_row2_col1': '86b03fbbfdda5a21b0fc5f49a70bf525',
        'AMBER_sample_grid_2x2_row2_col2': '6f9d3894b239f53ea0c50ceb966d066c',
        'AMBER_sample_grid_3x3_row1_col1': 'a3ca1d508f1bc3b69d4c9c4dc75302b6',
        'AMBER_sample_grid_3x3_row1_col2': 'bf3f18592e8e087fa18b92ca0aa4cb4b',
        'AMBER_sample_grid_3x3_row1_col3': 'd0429d05af7d819f4bdcb998b86c3c9e',
        'AMBER_sample_grid_3x3_row2_col1': '1dba8c7aad02e56a59e2425df0ced7e9',
        'AMBER_sample_grid_3x3_row2_col2': '99a05104cae80d6cacb9b8b07a956518',
        'AMBER_sample_grid_3x3_row2_col3': '1c2d2d9f14411e9eada9c9e043634e07',
        'AMBER_sample_grid_3x3_row3_col1': 'fca3fadec9488b54619cc13e5b3ff853',
        'AMBER_sample_grid_3x3_row3_col2': '532e0b9584b1aad983d68a874214d972',
        'AMBER_sample_grid_3x3_row3_col3': 'f162a99386c8533f38f93f56dcb6934b',
    }

    # It returns a dataframe
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.yorn import YOrN_Extraction, YOrN_auxeval
        from .utils.yorn import default_rating, MME_rating, Hallusion_rating, POPE_rating, AMBER_rating

        dataset = self.dataset_name
        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        storage = eval_file.replace('.xlsx', '_auxmatch.xlsx')
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            ans_map = {k: YOrN_Extraction(v) for k, v in zip(data['index'], data['prediction'])}
            if osp.exists(tmp_file):
                tmp = load(tmp_file)
                for k in tmp:
                    if ans_map[k] == 'Unknown' and tmp[k] != 'Unknown':
                        ans_map[k] = tmp[k]

            data['extracted'] = [ans_map[x] for x in data['index']]
            unknown = data[data['extracted'] == 'Unknown']

            model = judge_kwargs.get('model', 'exact_matching')
            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                model = None
                warnings.warn('OPENAI_API_KEY is not working properly, will use exact matching for evaluation')

            if model is not None:
                lt = len(unknown)
                lines = [unknown.iloc[i] for i in range(lt)]
                tups = [(model, line) for line in lines]
                indices = list(unknown['index'])
                if len(tups):
                    res = track_progress_rich(
                        YOrN_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            data['extracted'] = [ans_map[x] for x in data['index']]
            dump(data, storage)

        data = load(storage)
        if listinstr(['AMBER'], dataset):
            data['score'] = (data['answer'].str.lower() == data['extracted'].str.lower())
        else:
            data['score'] = (data['answer'] == data['extracted'])
        dump(data, storage)

        if dataset is not None and listinstr(['MME'], dataset):
            score = MME_rating(storage)
        elif dataset is not None and listinstr(['Hallusion'], dataset):
            score = Hallusion_rating(storage)
        elif dataset is not None and listinstr(['POPE'], dataset):
            score = POPE_rating(storage)
        elif dataset is not None and listinstr(['AMBER'], dataset):
            score = AMBER_rating(storage)
        else:
            score = default_rating(storage)

        score_tgt = eval_file.replace('.xlsx', '_score.csv')
        dump(score, score_tgt)
        return score
