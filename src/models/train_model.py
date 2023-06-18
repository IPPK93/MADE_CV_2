from typing import List, Dict

from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as f


def train_model_ctc(
        model: nn.Module,
        id_2_char: Dict[int, str],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.modules.loss._Loss,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        use_tensorboard: bool = True,
        show_sample_predicts: bool = False,
        pad_token: str = '\\',
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> Dict[str, List[str]]:
    losses = {'train': [], 'test': []}
    
    test_images, test_targets, test_lengths = [elem.to(device, ) for elem in next(iter(test_loader))]
    if show_sample_predicts:
        true_subsample = [[id_2_char[elem.item()] for elem in target] for target in test_targets[:10]]

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for train_data in train_loader:

            images, targets, target_lengths = [elem.to(device, non_blocking=True) for elem in train_data]

            logits = model(images)

            # recall that dims are (width, bsize, num_classes)
            log_probas = f.log_softmax(logits, dim=2)

            input_lengths = torch.tensor([logits.size(0)] * logits.size(1), dtype=torch.int32)

            loss = criterion(log_probas, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses['train'].append(loss.item())

        model.eval()
        with torch.no_grad():

            logits = model(test_images)

            # recall that dims are (width, bsize, num_classes)
            log_probas = f.log_softmax(logits, dim=2)

            input_lengths = torch.tensor([logits.size(0)] * logits.size(1), dtype=torch.int32)
            loss = criterion(log_probas, test_targets, input_lengths, test_lengths)

            if show_sample_predicts:
                pred = [
                    [id_2_char[s.item()] for s in ss if id_2_char[s.item()] != pad_token]
                    for ss in torch.argmax(log_probas.permute((1, 0, 2))[:10], axis=2)
                ]
                print(*zip(pred, true_subsample), sep='\n')

            losses['test'].append(loss.item())

        if scheduler is not None:
            scheduler.step()

        for title in ['train', 'test']:
            # ax.cla()
            # ax.set_title(f'{title} CTC')
            # ax.plot(losses[title], label=f"last_loss: {losses[title][-1]:.5f}")
            # ax.legend()
            break

        
    return losses