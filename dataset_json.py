import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


class DatasetJSON(Dataset):
    def __init__(self, data, sp_processor, max_length):
        if isinstance(data, list):
            lines = data
        elif isinstance(data, str):
            with open(data, encoding="UTF-8") as f:
                lines = f.readlines()
        else:
            raise TypeError("Only filename as String and lists are appropriate as data to dataloader")

        self.sp_processor = sp_processor
        self.unk_id = sp_processor.unk_id()
        self.bos_id = sp_processor.bos_id()
        self.eos_id = sp_processor.eos_id()
        self.pad_id = sp_processor.pad_id()
        self.lines = self._tokenize(lines)
        self.max_length = max_length

    def _tokenize(self, lines):
        tokenized_lines = []
        for i in tqdm(range(0, len(lines)), desc='Tokenizing data'):
            json_data = json.loads(lines[i])
            text = json_data["text"]
            result = [self.bos_id]
            result.extend(self.sp_processor.encode_as_ids(text))
            result.append(self.eos_id)
            if self.unk_id in result:
                continue
            tokenized_lines.append(result)
        return tokenized_lines

    def __getitem__(self, idx):
        return self.lines[idx][:-1], self.lines[idx][1:]

    def __len__(self):
        return len(self.lines)

    def collate_fn(self, batch):
        inputs = [torch.LongTensor(item[0]) for item in batch]
        targets = [torch.LongTensor(item[1]) for item in batch]

        # Pad sequences with 0s so they have the same length
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=self.pad_id)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=self.pad_id)
        inputs_truncated = inputs_padded[:, : self.max_length - 2]
        targets_truncated = targets_padded[:, : self.max_length - 2]
        padding_mask = torch.ones_like(inputs_truncated).float()
        padding_mask[inputs_truncated == self.pad_id] = float('-inf')
        padding_mask[inputs_truncated != self.pad_id] = padding_mask[inputs_truncated != self.pad_id] - 1
        return inputs_truncated, targets_truncated, padding_mask
