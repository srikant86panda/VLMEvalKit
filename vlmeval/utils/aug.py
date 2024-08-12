import yaml
from albumentations import *
from albumentations import Compose

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_augmentations(config):
    augmentations = []
    if 'individual_augmentations' in config:
        for aug in config['individual_augmentations']:
            aug_name = aug['name']
            aug_params = aug['params']
            augmentation = globals()[aug_name](**aug_params)
            augmentations.append({'name':aug_name, 'augmentation': augmentation})

    aug_list = []
    if 'multiple_augmentations' in config:
        for aug in config['multiple_augmentations']:
            aug_name = aug['name']
            aug_params = aug['params']
            augmentation = globals()[aug_name](**aug_params)
            aug_list.append(augmentation)
        augmentations.append({'name':'Compose', 'augmentation': Compose(aug_list)})
    return augmentations
