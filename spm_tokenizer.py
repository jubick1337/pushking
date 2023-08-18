import argparse
import json
import os
from tempfile import NamedTemporaryFile

import sentencepiece as spm

parser = argparse.ArgumentParser(
    prog="Sentencepiece tokenizer", description="This module is used to train sentencepiece tokenizer"
)
parser.add_argument(
    "--file_names",
    type=str,
    nargs='+',
    help="paths to source data which tokenizer will be trained on",
    required=True,
)
parser.add_argument(
    "--artifacts_dir", type=str, help="path to store trained model and vocab", default="spm_artifacts", required=False
)
parser.add_argument("--vocab_size", type=int, help="size of vocabulary of the model", default=512, required=False)
parser.add_argument(
    "--model_type", type=str, help="type of the algorythm to train model", default="bpe", required=False
)
parser.add_argument("--pad_id", type=int, default=0, required=False)
parser.add_argument("--unk_id", type=int, default=1, required=False)
parser.add_argument("--bos_id", type=int, default=2, required=False)
parser.add_argument("--eos_id", type=int, default=3, required=False)

args = parser.parse_args()

os.makedirs(args.artifacts_dir, exist_ok=True)

# Use NamedTemporaryFile instead of TemporaryFile
with NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as temp_file:
    for file_name in args.file_names:
        with open(file_name, "r", encoding="utf-8") as json_file:
            for line in json_file:
                data = json.loads(line)
                if "text" in data:
                    # Replace newline characters with a special token
                    modified_text = data["text"].replace("\n", " <n> ")
                    temp_file.write(modified_text + "\n")

    temp_file_name = temp_file.name

    SP_MODEL_PREFIX = f"{args.artifacts_dir}/sp_{args.model_type}_{args.vocab_size}"

    sp_train_command = (
        f"--input={temp_file_name} --model_prefix={SP_MODEL_PREFIX}"
        f" --vocab_size={args.vocab_size} --model_type={args.model_type}"
        f" --pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id}"
        f" --eos_id={args.eos_id} --character_coverage=1.0 --user_defined_symbols='<n>'"
    )

    spm.SentencePieceTrainer.train(sp_train_command)
