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


from fastapi import FastAPI
from pydantic import BaseModel

from paddlenlp import Taskflow

app = FastAPI()


chatbot = Taskflow("text2text_generation", batch_size=2)


class Item(BaseModel):
    prompt: str = "您好"
    decode_strategy: str = "sampling"
    top_k: int = 1
    top_p: float = 1.0
    temperature: float = 1.0
    tgt_length: int = 128
    max_seq_length: int = 128
    num_return_sequences: int = 1


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/text2text_generation/")
async def read_item(item: Item):
    message = item.prompt
    doc_dict = item.dict()
    chatbot.set_argument(doc_dict)
    results = chatbot(message)
    return results
