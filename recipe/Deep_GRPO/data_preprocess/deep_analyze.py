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
from datasets import load_dataset


DATA_TASK_FILE = "/data/download/DataScience-Instruct-500K/RL/datatask.parquet"
QA_TASK_FILE = "/data/download/DataScience-Instruct-500K/RL/qa.parquet"
RESEARCH_TASK_FILE = "/data/download/DataScience-Instruct-500K/RL/reseach.parquet"
WORKSPACE_ROOT_DIR = "/data/download/DataScience-Instruct-500K/RL/data"
SAVE_PATH = "/data/hf-datasets/data_analysis"

data_task_dataset = load_dataset("parquet", data_files=DATA_TASK_FILE)["train"]
qa_task_dataset = load_dataset("parquet", data_files=QA_TASK_FILE)["train"]
research_task_dataset = load_dataset("parquet", data_files=RESEARCH_TASK_FILE)["train"]


def process_fn(example, idx):
    prompt = example.pop("prompt")
    data_source = example.pop("data_source")
    workspace = example.pop("workspace_id")
    instruction = example.pop("input_seq")
    if data_source == "datatask":
      workspace = os.path.join(WORKSPACE_ROOT_DIR, "files", workspace)
    elif data_source == "research":
       workspace = os.path.join(WORKSPACE_ROOT_DIR, "bird", workspace)
    reward_spec = example.pop("reward_spec")
    ground_truth = reward_spec["ground_truth"]
    teacher_response = reward_spec["response"]
    if data_source == "datatask" or data_source == "research":
       agent_name = "data_analysis_agent_loop"
    else:
        agent_name = "reasoning_agent_loop"

    data = {
        "data_source": data_source,
        "prompt": prompt,
        "ability": data_source,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "agent_name": agent_name,
        "extra_info": {
            "index": idx,
            "instruction": instruction,
            "answer": teacher_response,
            "workspace": workspace,
        },
    }

    # if data_source == "research":
    #     data["reward_model"]["do_compare"] = True
    return data


original_data_columns = data_task_dataset.column_names
processed_data_task = data_task_dataset.map(
    process_fn,
    with_indices=True,
    remove_columns=original_data_columns
).filter(lambda x: x["reward_model"]["ground_truth"].strip())

original_qa_columns = qa_task_dataset.column_names
processed_qa_task = qa_task_dataset.map(
    process_fn,
    with_indices=True,
    remove_columns=original_qa_columns
).filter(lambda x: x["reward_model"]["ground_truth"].strip())

original_research_columns = research_task_dataset.column_names
processed_research_task = research_task_dataset.map(
    process_fn,
    with_indices=True,
    remove_columns=original_research_columns
).filter(lambda x: x["reward_model"]["ground_truth"].strip())

os.makedirs(SAVE_PATH, exist_ok=True)
processed_data_task.to_parquet(os.path.join(SAVE_PATH, "data_task.parquet"))
processed_qa_task.to_parquet(os.path.join(SAVE_PATH, "qa_task.parquet"))
processed_research_task.to_parquet(os.path.join(SAVE_PATH, "research_task.parquet"))