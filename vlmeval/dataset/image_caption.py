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
        'COCO_VAL_impulse_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_impulse_noise_1.tsv',
        'COCO_VAL_sample_brightness_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_brightness_1.tsv',
        'COCO_VAL_sample_brightness_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_brightness_5.tsv',
        'COCO_VAL_sample_contrast_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_contrast_1.tsv',
        'COCO_VAL_sample_contrast_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_contrast_5.tsv',
        'COCO_VAL_sample_defocus_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_defocus_blur_1.tsv',
        'COCO_VAL_sample_defocus_blur_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_defocus_blur_5.tsv',
        'COCO_VAL_sample_elastic_transform_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_elastic_transform_1.tsv',
        'COCO_VAL_sample_elastic_transform_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_elastic_transform_5.tsv',
        'COCO_VAL_sample_fog_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_fog_1.tsv',
        'COCO_VAL_sample_fog_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_fog_5.tsv',
        'COCO_VAL_sample_frost_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_frost_1.tsv',
        'COCO_VAL_sample_frost_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_frost_5.tsv',
        'COCO_VAL_sample_gaussian_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_gaussian_noise_1.tsv',
        'COCO_VAL_sample_gaussian_noise_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_gaussian_noise_5.tsv',
        'COCO_VAL_sample_glass_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_glass_blur_1.tsv',
        'COCO_VAL_sample_glass_blur_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_glass_blur_5.tsv',
        'COCO_VAL_sample_impulse_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_impulse_noise_1.tsv',
        'COCO_VAL_sample_impulse_noise_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_impulse_noise_5.tsv',
        'COCO_VAL_sample_jpeg_compression_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_jpeg_compression_1.tsv',
        'COCO_VAL_sample_jpeg_compression_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_jpeg_compression_5.tsv',
        'COCO_VAL_sample_motion_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_motion_blur_1.tsv',
        'COCO_VAL_sample_motion_blur_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_motion_blur_5.tsv',
        'COCO_VAL_sample_pixelate_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_pixelate_1.tsv',
        'COCO_VAL_sample_pixelate_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_pixelate_5.tsv',
        'COCO_VAL_sample_shot_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_shot_noise_1.tsv',
        'COCO_VAL_sample_shot_noise_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_shot_noise_5.tsv',
        'COCO_VAL_sample_snow_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_snow_1.tsv',
        'COCO_VAL_sample_snow_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_snow_5.tsv',
        'COCO_VAL_sample_speckle_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_speckle_noise_1.tsv',
        'COCO_VAL_sample_speckle_noise_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_speckle_noise_5.tsv',
        'COCO_VAL_sample': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample.tsv',
        'COCO_VAL_sample_zoom_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_zoom_blur_1.tsv',
        'COCO_VAL_sample_zoom_blur_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_zoom_blur_5.tsv',
        'COCO_VAL_snow_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_snow_1.tsv',
        'COCO_VAL_zoom_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_zoom_blur_1.tsv'
    }

    DATASET_MD5 = {
        'COCO_VAL': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_impulse_noise_1': 'd8368331e2ebe19246fbca069f2801fb',
        'COCO_VAL_sample_brightness_1': 'afe5045e55604ef3e95328a91e187142',
        'COCO_VAL_sample_brightness_5': '595fb314cc9f167730fe02f67a4139ff',
        'COCO_VAL_sample_contrast_1': '8708643b751198974669a67ade086bbe',
        'COCO_VAL_sample_contrast_5': '7268a590a70f64b30885a609eb2b585f',
        'COCO_VAL_sample_defocus_blur_1': '908d697b8eab7e7c581213ea653dbd06',
        'COCO_VAL_sample_defocus_blur_5': '98db6aadbbcba148075f5631a9746638',
        'COCO_VAL_sample_elastic_transform_1': '25b70f15e873ced5331873ed34021bd5',
        'COCO_VAL_sample_elastic_transform_5': 'd52f8111f99a5e5e01fb5dc97d06484c',
        'COCO_VAL_sample_fog_1': '91bfcf86a484f309b28d1572aa2d6d2d',
        'COCO_VAL_sample_fog_5': '631310c7ef09ef2f0edf6aeb813ef683',
        'COCO_VAL_sample_frost_1': 'e0287002b89c9c91b8352efbb0a82a76',
        'COCO_VAL_sample_frost_5': 'f67015727f82bbff54a67ea4b1513b0e',
        'COCO_VAL_sample_gaussian_noise_1': 'a84164d58bc588e759aa27593d4d0fb2',
        'COCO_VAL_sample_gaussian_noise_5': '25fd3ea2208c24d5b6f73c392e7e15d5',
        'COCO_VAL_sample_glass_blur_1': 'ad78bd7a7e701b796df07327fc57dc30',
        'COCO_VAL_sample_glass_blur_5': '58e56dc44df17006338ea9781a218a87',
        'COCO_VAL_sample_impulse_noise_1': 'd7886b1ed0880779dc57197ea24ced9d',
        'COCO_VAL_sample_impulse_noise_5': '403cfd2b0fb9b495e60a757d7b40cce9',
        'COCO_VAL_sample_jpeg_compression_1': 'ac225483c47d1be82f6428decff242e7',
        'COCO_VAL_sample_jpeg_compression_5': 'a6f7141b83146b35e66d4dad31d6d4a1',
        'COCO_VAL_sample_motion_blur_1': '27f9c3aa87d860cabd064a3b98c36457',
        'COCO_VAL_sample_motion_blur_5': '62c33b9d56aa81112fa2238d01fdfa7f',
        'COCO_VAL_sample_pixelate_1': '24967cd9b4bc3d75ced4c7df522df9a7',
        'COCO_VAL_sample_pixelate_5': 'fa153e62bbb0b5d0d765dada846c98a9',
        'COCO_VAL_sample_shot_noise_1': '8d89dd90137fb6652cd979e7412764b7',
        'COCO_VAL_sample_shot_noise_5': '54a47f14b249f7fbe2d860ad6f4ad80e',
        'COCO_VAL_sample_snow_1': '9ba680cfab0d78639b7e14abbe52563e',
        'COCO_VAL_sample_snow_5': '4e161a36f2a54abf6e5441979cdbedf2',
        'COCO_VAL_sample_speckle_noise_1': 'd03f45546e80e9dcaa02a4ccf3429a9a',
        'COCO_VAL_sample_speckle_noise_5': '72e1104c80e6e9912fda3604886a5d31',
        'COCO_VAL_sample': '5b6ed6e5f35024d003804372a13533c4',
        'COCO_VAL_sample_zoom_blur_1': '9cc2d7886e31176cb9b9875da6cf76ac',
        'COCO_VAL_sample_zoom_blur_5': 'e15808045424cf1ed9794788711066c6',
        'COCO_VAL_snow_1': 'feb4676bd65691aada6987390bb5a7ff',
        'COCO_VAL_zoom_blur_1': '5efa26732ed37820708ddb3782e2950d',
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
