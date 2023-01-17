# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import base64
import json
import logging
import os
import pickle
from dataclasses import dataclass
from io import BytesIO
from math import ceil
from pathlib import Path

import lmdb
import numpy as np
import paddle
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from paddle.vision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from PIL import Image


def _convert_to_rgb(image):
    return image.convert("RGB")


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", '"').replace("”", '"')
    return text


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224, tokenizer=None):
        self.lmdb_path = lmdb_path

        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)
        lmdb_pairs = os.path.join(lmdb_path, "pairs")
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(
            lmdb_pairs, split
        )
        lmdb_imgs = os.path.join(lmdb_path, "imgs")
        assert os.path.isdir(lmdb_imgs), "The LMDB directory {} of {} image base64 strings does not exist!".format(
            lmdb_imgs, split
        )

        # open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)

        # fetch number of pairs and images
        self.number_samples = int(self.txn_pairs.get(key=b"num_samples").tobytes().decode("utf-8"))
        self.number_images = int(self.txn_imgs.get(key=b"num_images").tobytes().decode("utf-8"))
        logging.info(
            "{} LMDB file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples)
        )

        super(LMDBDataset, self).__init__()

        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        self.global_batch_size = 1  # will be modified to the exact global_batch_size after calling pad_dataset()

        self.split = split
        self.max_txt_length = max_txt_length

        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)
        self.tokenizer = tokenizer

    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = Compose(
                [
                    RandomResizedCrop(resolution, scale=(0.9, 1.0), interpolation="bicubic"),
                    RandomHorizontalFlip(0.5),
                    _convert_to_rgb,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
        else:
            transform = Compose(
                [
                    Resize((resolution, resolution), interpolation="bicubic"),
                    _convert_to_rgb,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
        return transform

    def __del__(self):
        if hasattr(self, "env_pairs"):
            self.env_pairs.close()
        if hasattr(self, "env_imgs"):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.number_samples

        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode("utf-8")).tobytes())
        image_id, text_id, raw_text = pair

        image_b64 = self.txn_imgs.get("{}".format(image_id).encode("utf-8")).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image = self.transform(image)
        # text = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        texts = self.tokenizer([_preprocess_text(raw_text)], max_len=self.max_txt_length, padding="max_length")
        # print(text['input_ids'][0])
        text = texts["input_ids"][0]
        # text = tokenizer([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        # eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        eos_index = text.index(self.tokenizer.vocab["[SEP]"])
        eos_index = np.array(eos_index)
        return {"pixel_values": image, "input_ids": text, "index": eos_index}


def pad_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, "r") as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedBatchSampler
    dataset: LMDBDataset
    epoch_id: int


def get_eval_txt_dataset(args, max_txt_length=24, tokenizer=None):
    input_filename = args.text_data
    dataset = EvalTxtDataset(input_filename, max_txt_length=max_txt_length, tokenizer=tokenizer)
    return dataset


def get_eval_img_dataset(args):
    lmdb_imgs = args.image_data
    dataset = EvalImgDataset(lmdb_imgs, resolution=224)
    return dataset


def get_train_eval_dataset(args, epoch_id=0, max_txt_length=64, tokenizer=None):
    train_dataset = LMDBDataset(
        args.train_data,
        split="train",
        max_txt_length=max_txt_length,
        tokenizer=tokenizer,
        use_augment=True,
        resolution=224,
    )
    eval_dataset = LMDBDataset(
        args.val_data,
        split="val",
        max_txt_length=max_txt_length,
        tokenizer=tokenizer,
        use_augment=False,
        resolution=224,
    )
    return train_dataset, eval_dataset


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    dataset = LMDBDataset(
        db_path,
        split="train" if is_train else "val",
        max_txt_length=max_txt_length,
        use_augment=args.use_augment if is_train else False,
        resolution=fetch_resolution(args.vision_model),
    )
    print(dataset[0])

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size if is_train else args.valid_batch_size
    # global_batch_size = batch_size * paddle.distributed.get_world_size()
    global_batch_size = batch_size
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs).
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
    # sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, drop_last=False)
    # sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        # pin_memory=False,
        num_workers=args.num_workers if is_train else 1,
        # sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(args, is_train=True, max_txt_length=max_txt_length, epoch_id=epoch_id)

    if args.val_data:
        data["val"] = get_dataset(args, is_train=False, max_txt_length=max_txt_length, epoch_id=epoch_id)

    return data


def create_dataloader(dataset, mode="train", batch_size=1, num_workers=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=batchify_fn, return_list=True
    )


class EvalTxtDataset(Dataset):
    def __init__(self, jsonl_filename, max_txt_length=24, tokenizer=None):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)

        logging.debug(f"Loading jsonl data from {jsonl_filename}.")
        self.texts = []
        with open(jsonl_filename, "r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                text_id = obj["text_id"]
                text = obj["text"]
                self.texts.append((text_id, text))
        logging.debug(f"Finished loading jsonl data from {jsonl_filename}.")

        self.max_txt_length = max_txt_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_id, text = self.texts[idx]
        # text = tokenize([_preprocess_text(str(text))], context_length=self.max_txt_length)[0]
        texts = self.tokenizer([_preprocess_text(str(text))], max_len=self.max_txt_length, padding="max_length")
        # print(text['input_ids'][0])
        text = texts["input_ids"][0]
        return {"text_id": text_id, "input_ids": text}


class EvalImgDataset(Dataset):
    def __init__(self, lmdb_imgs, resolution=224):
        assert os.path.isdir(lmdb_imgs), "The image LMDB directory {} not exists!".format(lmdb_imgs)

        logging.debug(f"Loading image LMDB from {lmdb_imgs}.")

        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)
        self.cursor_imgs = self.txn_imgs.cursor()
        self.iter_imgs = iter(self.cursor_imgs)
        self.number_images = int(self.txn_imgs.get(key=b"num_images").tobytes().decode("utf-8"))
        logging.info("The specified LMDB directory contains {} images.".format(self.number_images))

        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        return Compose(
            [
                Resize((resolution, resolution), interpolation="bicubic"),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        img_id, image_b64 = next(self.iter_imgs)
        if img_id == b"num_images":
            img_id, image_b64 = next(self.iter_imgs)

        img_id = img_id.tobytes()
        image_b64 = image_b64.tobytes()

        img_id = int(img_id.decode(encoding="utf8", errors="ignore"))
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image = self.transform(image)

        return img_id, image
