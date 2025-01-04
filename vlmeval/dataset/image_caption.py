from .image_base import ImageBaseDataset
from ..smp import *


class COCO_Caption_Scorer():
    def __init__(self, ref, gt):
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print('%s: %0.3f' % (m, sc * 100))
                total_scores['Bleu'] = [x * 100 for x in score]
            else:
                print('%s: %0.3f' % (method, score * 100))
                total_scores[method] = score * 100

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores


class ImageCaptionDataset(ImageBaseDataset):

    TYPE = 'Caption'

    DATASET_URL = {
        'COCO_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL.tsv',
        'COCO_VAL_sample': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample.tsv'
        'COCO_VAL_sample_grid_2x1_row1_col1': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_2x1_row1_col1.tsv'
        'COCO_VAL_sample_grid_2x1_row2_col1': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_2x1_row2_col1.tsv'
        'COCO_VAL_sample_grid_1x2_row1_col1': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_1x2_row1_col1.tsv'
        'COCO_VAL_sample_grid_1x2_row1_col2': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_1x2_row1_col2.tsv'
        'COCO_VAL_sample_grid_2x2_row1_col1': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_2x2_row1_col1.tsv'
        'COCO_VAL_sample_grid_2x2_row1_col2': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_2x2_row1_col2.tsv'
        'COCO_VAL_sample_grid_2x2_row2_col1': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_2x2_row2_col1.tsv'
        'COCO_VAL_sample_grid_2x2_row2_col2': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_2x2_row2_col2.tsv'
        'COCO_VAL_sample_grid_3x3_row1_col1': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_3x3_row1_col1.tsv'
        'COCO_VAL_sample_grid_3x3_row1_col2': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_3x3_row1_col2.tsv'
        'COCO_VAL_sample_grid_3x3_row1_col3': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_3x3_row1_col3.tsv'
        'COCO_VAL_sample_grid_3x3_row2_col1': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_3x3_row2_col1.tsv'
        'COCO_VAL_sample_grid_3x3_row2_col2': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_3x3_row2_col2.tsv'
        'COCO_VAL_sample_grid_3x3_row2_col3': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_3x3_row2_col3.tsv'
        'COCO_VAL_sample_grid_3x3_row3_col1': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_3x3_row3_col1.tsv'
        'COCO_VAL_sample_grid_3x3_row3_col2': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_3x3_row3_col2.tsv'
        'COCO_VAL_sample_grid_3x3_row3_col3': 'https://huggingface.co/datasets/Srikant86/VLMEval/blob/main/COCO_VAL/COCO_VAL_sample_grid_3x3_row3_col3.tsv'
    }

    DATASET_MD5 = {
        'COCO_VAL': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample': '5b6ed6e5f35024d003804372a13533c4',
        'COCO_VAL_sample_grid_2x1_row1_col1': '9999f71c872ada7193ac6f96e9c4030e',
        'COCO_VAL_sample_grid_2x1_row2_col1': '18ec1a575214bab28a2e50003e957299',
        'COCO_VAL_sample_grid_1x2_row1_col1': '6b02299cf429ca0ad398d01603688448',
        'COCO_VAL_sample_grid_1x2_row1_col2': '9ebee03bedc40d91c2fd08ff3dee1102',
        'COCO_VAL_sample_grid_2x2_row1_col1': 'f5e674127cc699ebfb77f66ba5581868',
        'COCO_VAL_sample_grid_2x2_row1_col2': '4c9f1408805b62e02695efe5cd546a1b',
        'COCO_VAL_sample_grid_2x2_row2_col1': 'b6a16dc21248550c2c287ffc5a6adeb3',
        'COCO_VAL_sample_grid_2x2_row2_col2': '149df47164d35499bfbf36481b2e8f5c',
        'COCO_VAL_sample_grid_3x3_row1_col1': '7b5807ef64318afbe27858e48507eaae',
        'COCO_VAL_sample_grid_3x3_row1_col2': 'e1640bc9276cf806e80248169b90b3d4',
        'COCO_VAL_sample_grid_3x3_row1_col3': 'd12c40a377dcc8593ba1d745cd3867f7',
        'COCO_VAL_sample_grid_3x3_row2_col1': '0359f4323e287b33c06fde01d2d8ca39',
        'COCO_VAL_sample_grid_3x3_row2_col2': 'ffe43118269c9af4d703c0297a492053',
        'COCO_VAL_sample_grid_3x3_row2_col3': '56ef9a830da851d0d225e6363d4ebe47',
        'COCO_VAL_sample_grid_3x3_row3_col1': '3d81fb12514a28c410ba84f8b473087a',
        'COCO_VAL_sample_grid_3x3_row3_col2': '830196882c5b8774a32a4f8ead8c6370',
        'COCO_VAL_sample_grid_3x3_row3_col3': 'f5f4996a9e79d8c90708f937f16d7936',
    }

    def load_data(self, dataset):
        data = super().load_data(dataset)
        if 'question' not in data:
            data['question'] = [(
                'Please describe this image in general. Directly provide the description, '
                'do not include prefix like "This image depicts". '
            )] * len(data)
        return data

    # It returns a dictionary of scores
    @classmethod
    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        ref, gt = {}, {}
        for i, line in enumerate(lines):
            ref[str(i)] = [str(line['prediction'])]
            gt[str(i)] = eval(line['answer'])

        scorer = COCO_Caption_Scorer(ref, gt)
        coco_caption_score_dict = scorer.compute_scores()
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(coco_caption_score_dict, score_pth)
        return coco_caption_score_dict
