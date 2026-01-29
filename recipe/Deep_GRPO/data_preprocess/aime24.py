# Copyright 2024 Anonymous Authors
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
import os

import datasets

SAVE_PATH = "/data/hf-datasets/aime24"
TEST_DATASET_PATH = "/root/paddlejob/workspace/understand-r1-zero/datasets/evaluation_suite/aime"


test_dataset = datasets.load_from_disk(TEST_DATASET_PATH)
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."


def make_test_map_fn(split):
    def process_fn(example, idx):
        question = example["problem"]

        question = question + " " + instruction_following

        answer = example["answer"]

        if isinstance(answer, float) or isinstance(answer, int):
          answer = str(answer)

        if isinstance(answer, str):
            answer = [answer]    

        data = {
            "data_source": "AIME24",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "agent_name": "reasoning_agent_loop",
            "extra_info": {"split": split, "index": idx, "question": question, "answer": answer},
        }
        return data

    return process_fn

test_dataset = test_dataset.map(function=make_test_map_fn("test"), with_indices=True, remove_columns=test_dataset.column_names)

os.makedirs(SAVE_PATH, exist_ok=True)
test_dataset.to_parquet(os.path.join(SAVE_PATH, "test.parquet"))