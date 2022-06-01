# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

from tqdm import tqdm

import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import convert_example, get_id_to_label

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, default="checkpoints/model_900/model_state.pdparams", help="The path to model parameters to be loaded.")
parser.add_argument("--label_file", default="data/label_level1.txt", type=str, help="label file for classfication tasks.")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


@paddle.no_grad()
def predict(model, data, tokenizer, label_map, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text in data:
        # example = {"text": text}
        input_ids, token_type_ids = convert_example(
            text,
            tokenizer,
            max_seq_length=args.max_seq_length,
            is_test=True,
            is_pair=True)
        examples.append((input_ids, token_type_ids))

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): fn(samples)

    results = []
    model.eval()
    for batch in tqdm(batches):
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            yield {'text_a': data[0], 'text_b': data[1], 'label': data[2]}


if __name__ == "__main__":
    paddle.set_device(args.device)

    data = [
        {
            'text_a': '这个是一个西边间吗？西边间。对。',
            'text_b': '对。',
            'label': '样板间介绍'
        },
        {
            'text_a': '反正先选九号楼。九号楼你有什么想法吗？要。',
            'text_b': '要。',
            'label': '洽谈商议'
        },
        {
            'text_a':
            '第一大优势的话它是采取国际化莫兰迪高级灰外立面。简洁的线条加上大面的玻璃，给人非常现代的感觉。那么这种莫迪高级灰外立面呢是适用于国内外的一些豪宅，才会特别这种莫兰迪高级灰外立面。',
            'text_b': '那么这种莫迪高级灰外立面呢是适用于国内外的一些豪宅，才会特别这种莫兰迪高级灰外立面。',
            'label': '沙盘讲解'
        },
    ]

    label_map = get_id_to_label(args.label_file)

    model_name_or_path = 'ernie-3.0-base-zh'
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_classes=len(label_map))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = predict(
        model, data, tokenizer, label_map, batch_size=args.batch_size)
    with open('result.csv', 'w') as f:
        for idx, text in enumerate(data):
            f.write(str(text) + '\t' + results[idx] + '\n')

    for idx, text in enumerate(data):
        print('Data: {} \t Lable: {}'.format(text, results[idx]))
