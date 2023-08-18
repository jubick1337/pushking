from torch.utils.data import Dataset
from tqdm import tqdm


class DatasetNlfi(Dataset):
    def __init__(self, data, sp_processor, n_lines):
        if isinstance(data, list):
            lines = data
        elif isinstance(data, str):
            with open(data, encoding="UTF-8") as f:
                lines = f.readlines()
        else:
            raise TypeError("Only filename as String and lists are appropriate as data to dataloader")

        self.n_lines = n_lines
        self.sp_processor = sp_processor
        self.unk_id = sp_processor.unk_id()
        self.bos_id = sp_processor.bos_id()
        self.eos_id = sp_processor.eos_id()
        self.pad_id = sp_processor.pad_id()
        self.lines = self._tokenize(lines)

    def _tokenize(self, lines):
        tokenized_lines = []
        for i in tqdm(range(0, len(lines), self.n_lines), desc='Tokenizing data'):
            to_tokenize_lines = "".join(lines[i : i + self.n_lines])
            result = [self.bos_id]
            result.extend(self.sp_processor.encode_as_ids(to_tokenize_lines))
            result.append(self.eos_id)
            assert self.unk_id not in result
            tokenized_lines.append(result)
        return tokenized_lines

    def __getitem__(self, idx):
        return self.lines[idx][:-1], self.lines[idx][1:]

    def __len__(self):
        return len(self.lines)
