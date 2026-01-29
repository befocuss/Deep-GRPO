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
import logging
import os
from typing import Any, Dict
from uuid import uuid4

from recipe.DEEP_GRPO.protocol import FINISH_REASON, Node
from recipe.DEEP_GRPO.agent_loop.tree_search_agent_loop import TSAgentLoop


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ReasoningAgentLoop(TSAgentLoop):
    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)

    @classmethod
    def init_class(cls, config, tokenizer):
        if cls._class_initialized:
            return

        super().init_class(config, tokenizer)
        cls.tokens_per_node = config.actor_rollout_ref.rollout.spo.tokens_per_node

    async def _step(self, parent_node: Node, sampling_params: Dict[str, Any], request_id: str) -> Node:
        prompt_ids = parent_node.prompt_ids + parent_node.response_ids
        
        remaining_len = self.max_model_len - len(prompt_ids)
        assert remaining_len > 0
        tokens_to_generate = min(self.tokens_per_node, remaining_len)
        
        sampling_params_for_step = sampling_params.copy()
        sampling_params_for_step['max_tokens'] = tokens_to_generate

        assert len(prompt_ids) < self.max_model_len
        result = await self.server_manager.generate_with_finish_reason(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params_for_step
        )
        response_ids = result["token_ids"]
        response_mask = [1] * len(response_ids)

        if result["finish_reason"] == "length":
            if len(prompt_ids) + len(response_ids) >= self.max_model_len:
                finish_reason = FINISH_REASON.EXCEED_LENGTH
            else:
                finish_reason = FINISH_REASON.STOP
        elif result["finish_reason"] == "stop":
            finish_reason = FINISH_REASON.COMPLETED
        else:
            raise RuntimeError(f"Unexpected finish_reason from server: {result['finish_reason']}")

        return Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            data_instance=parent_node.data_instance,
            finish_reason=finish_reason,
            num_turns=parent_node.num_turns + 1
        )