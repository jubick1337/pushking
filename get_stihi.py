import json
import re

from datasets import load_dataset
from tqdm import tqdm

MAX_LEN = 50000
pattern = re.compile(r'[\U00010000-\U0010ffff]|[\u4e00-\u9fff]|[\u2600-\u26FF\u2700-\u27BF]')

dataset = load_dataset('IlyaGusev/stihi_ru', split="train", streaming=True)
with open('materials/stihi.json', 'w', encoding='utf-8') as f:
    i = 0
    for sample in tqdm(dataset):
        text = sample['text']
        if len(text.split(' ')) > 10 and not pattern.search(text):
            json.dump({'text': text}, f, ensure_ascii=False)
            f.write('\n')
            i += 1
        if i == MAX_LEN:
            break
