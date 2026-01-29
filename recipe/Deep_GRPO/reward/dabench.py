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
import re

import logging
import os

from recipe.Deep_GRPO.protocol import RewardInfo


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def extract_format(input_string):
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    answer_names = [match[0] for match in matches]
    answers = [match[1] for match in matches]
    return answer_names, answers


def is_equal(response, label):
    if response is None:
        return False
    response = response.strip().strip('"').strip("'")
    label = label.strip().strip('"').strip("'")
    if response == label:
        return True
    else:
        try:
            return abs(float(response) - float(label)) < 1e-6
        except:
            return False

def extract_answer(text):
    matches = re.findall(r"<Answer>(.*?)</Answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return None


def compute_score(chat_history_str: str, ground_truth) -> RewardInfo:
    answer_str = extract_answer(chat_history_str)
    if answer_str is None:
        return RewardInfo(reward=0,
                          completed=0)

    label_answers = {ans[0]: ans[1] for ans in ground_truth}
    answer_names, answers = extract_format(answer_str)
    extracted_answers = dict(zip(answer_names, answers))
    completed = 1
    for anser_name in label_answers:
        if extracted_answers.get(anser_name) is None:
            completed = 0
            break
    correct_answers = {anser_name: is_equal(extracted_answers.get(anser_name), label_answers[anser_name]) for anser_name in label_answers.keys()}
    correctness_reward = 1 if all(correct_answers.values()) else 0
    return RewardInfo(reward=correctness_reward,
                      completed=completed)