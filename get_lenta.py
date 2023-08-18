import json
import re

from corus import load_lenta
from tqdm import tqdm

MAX_LEN = 25000
lenta = load_lenta('lenta-ru-news.csv.gz')

# Define a regular expression pattern to match emojis and Chinese characters
pattern = re.compile(r'[\U00010000-\U0010ffff]|[\u4e00-\u9fff]|[\u2600-\u26FF\u2700-\u27BF]')

with open('materials/lenta.json', 'w', encoding='utf-8') as f:
    i = 0
    for data in tqdm(lenta):
        text = data.text.replace('\xa0', ' ')
        if len(text.split(' ')) > 10 and not pattern.search(text):
            json.dump({'text': text}, f, ensure_ascii=False)
            f.write('\n')
            i += 1
        if i == MAX_LEN:
            break
