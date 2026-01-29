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
import logging
from recipe.Deep_GRPO.protocol import RewardInfo
from recipe.Deep_GRPO.reward.math_grader import boxed_reward_fn


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_score(chat_history_str, ground_truth) -> RewardInfo:
    info, reward = boxed_reward_fn(chat_history_str, ground_truth, fast=False)
    completed = int(info["formatted"])
    return RewardInfo(reward=reward, completed=completed)