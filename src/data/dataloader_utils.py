from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(
        data: List[Tuple[torch.Tensor, torch.Tensor, int]],
        padding_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, labels, lengths = zip(*data)
    images = torch.cat(images)
    
    labels = pad_sequence(labels, batch_first=True, padding_value=padding_value)
    
    return images, labels, torch.tensor(lengths, dtype=torch.int32)