import os
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image

import pandas as pd
from collections import Counter


class OCRTrainDataset(Dataset):
    def __init__(
        self,
        root_dir: str = os.path.join('data', 'raw', 'vk-made-ocr'),
        desired_width: int = 80,
        desired_height: int = 32,
        pad_token: str = '\\',
        vocab_size: int = 180,
        limit: int = -1,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.path = Path(os.path.join(root_dir, 'train', 'train'))
        self.width = desired_width
        self.height = desired_height
        self.limit = limit
        self.labels = pd.read_csv(os.path.join(root_dir, 'train_labels.csv')).fillna('')
        
        self.pad_token = pad_token
        self.vocab_size = vocab_size
        self.id_2_char = None
        self.char_2_id = None
        
        chars_in_dataset = Counter(''.join(self.labels['Expected'].values))
        self.fill_vocabs(chars_in_dataset)
        self.clean_labels(chars_in_dataset)
        
        if self.limit != -1:
            self.labels = self.labels.iloc[:self.limit]
        
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path = self.path / self.labels.iloc[idx]['Id']
        
        with Image.open(path) as image_pil:
            image = image_pil \
                if image_pil.size[0] / image_pil.size[1] > 0.8 \
                else image_pil.transpose(Image.Transpose.ROTATE_90)
            image = image.resize((self.width, self.height))
            image = torch.from_numpy(np.array(image).astype(np.float32) / 255).unsqueeze(0)
        
        image = image.permute((0, 3, 1, 2))
        
        target = [self.char_2_id[x] for x in self.labels.iloc[idx]['Expected']]
        target = torch.tensor(target, dtype=torch.int32)
        
        return image, target, len(target)
    
    def fill_vocabs(self, chars_in_dataset: Counter) -> None:
        '''
        Fill (id, char) and (char, id) vocabs for current dataset.
        '''
        self.id_2_char = dict(enumerate(map(lambda x: x[0], chars_in_dataset.most_common(self.vocab_size)), start=1))
        self.id_2_char[0] = self.pad_token
        self.char_2_id = {char: label for label, char in self.id_2_char.items()}
        
    def clean_labels(self, chars_in_dataset: Counter) -> None:
        '''
        Remove Out Of Vocabulary points ((image, label) pairs) from dataset
        '''
        out_of_dict_elems = set([elem for elem in chars_in_dataset.keys() if elem not in self.char_2_id])

        self.labels = self.labels[~(self.labels['Expected']
                        .map(out_of_dict_elems.intersection)
                        .map(len) > 0)].reset_index(drop=True)