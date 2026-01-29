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
def show_ground_truth(node):
  return node.non_tensor_batch["reward_model"]["ground_truth"]

def show_whole_trajectory(node):
  return self.tokenizer.decode(node.batch["input_ids"][node.batch["attention_mask"].bool()], skip_special_tokens=False)

def show_node_advantage(node):
  _min = node.batch["advantages"].min()
  _max = node.batch["advantages"].max()
  if _min < 0:
    return _min
  if _max > 0:
    return _max
  return 0

def show_node_reward(node):
  return node.batch["token_level_scores"].sum()

def get_nodes_within_same_tree(node):
  nodes = []
  tree_id = node.non_tensor_batch["__tree_ids__"]
  for item in batch:
    if item.non_tensor_batch["__tree_ids__"] == tree_id:
      nodes.append(item)
  return nodes

def find_root(node_list):
  class Node:
    def __init__(self):
      self.children = []

  root = None

  node_id_map = {}
  for node in node_list:
    node_id = node.non_tensor_batch["__node_ids__"]
    node_id_map[node_id] = node
  
  for node in node_list:
    node.children = []

  for node in node_list:
    parent_node_id = node.non_tensor_batch["__parent_node_ids__"]
    assert parent_node_id
    if parent_node_id not in node_id_map:
      root = Node()
      node_id_map[parent_node_id] = root
    if parent_node_id in node_id_map:
      parent = node_id_map[parent_node_id]
      parent.children.append(node)

  return root


MODEL = "/data/hf-models/Qwen2.5-1.5B-Instruct"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokenizer.decode(batch.batch[767]["input_ids"][batch.batch[767]["attention_mask"].bool()], skip_special_tokens=False)