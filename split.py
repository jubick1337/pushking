import json
import random

# Set the random seed for reproducibility
random.seed(42)

# Define the input file and output files
input_file = 'materials/pretrain.json'
train_file = 'materials/train.json'
val_file = 'materials/val.json'

# Define the proportion for the training set
train_proportion = 0.95

# Load the data from the input file
with open(input_file, 'r', encoding='utf-8') as f:
    data = [line for line in f]

# Shuffle the data
random.shuffle(data)

# Split the data into training and validation sets
train_size = int(len(data) * train_proportion)
train_data = data[:train_size]
val_data = data[train_size:]


# Replace \n with <n> in the text field of each JSON entry
def replace_newlines(lines):
    modified_lines = []
    for line in lines:
        entry = json.loads(line)
        if 'text' in entry:
            entry['text'] = entry['text'].replace('\n', '<n>')
        modified_lines.append(json.dumps(entry, ensure_ascii=False) + '\n')
    return modified_lines


train_data = replace_newlines(train_data)
val_data = replace_newlines(val_data)

# Write the training data to the train file
with open(train_file, 'w', encoding='utf-8') as f:
    for line in train_data:
        f.write(line)

# Write the validation data to the val file
with open(val_file, 'w', encoding='utf-8') as f:
    for line in val_data:
        f.write(line)
