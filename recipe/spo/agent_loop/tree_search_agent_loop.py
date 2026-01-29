from abc import abstractmethod
from typing import Any, Dict, List, Union, Tuple, Optional

import asyncio
import logging
import os
from uuid import uuid4
import re
import json

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from recipe.spo.protocol import Node, FINISH_REASON, TSTrainAgentLoopOutput, TSValAgentLoopOutput, HardFailureSeed
from recipe.spo.reward.reward_manager import score_node
from recipe.spo.utils import call_teacher_with_retry
from recipe.spo.prompts.tree_search import TEACHER_SELECTION_PROMPT_TEMPLATE
from recipe.spo.branching_strategy import RandomBranchingStrategy, UtilitySamplingStrategy, BranchingSelection, BranchingFeedback

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _parse_teacher_selection_reply(reply: str) -> int:
    matches = re.findall(r"```(?:\w+)?\s*(\{.*?\})\s*```", reply, re.DOTALL)
    if not matches:
        matches = re.findall(r"```(?:\w+)?\s*(.*?)```", reply, re.DOTALL)
    if not matches:
        matches = [reply]
    for json_str in reversed(matches):
        try:
            cleaned_str = json_str.strip()
            data = json.loads(cleaned_str)
            required_key = "first_error_step_index"
            if required_key in data:
                return int(data[required_key])
                
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    raise ValueError(f"Failed to parse teacher selection JSON from reply. Reply:\n{reply}")


class TSAgentLoop(AgentLoopBase):
    
    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)

    @classmethod
    def init_class(cls, config, tokenizer):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.tokenizer = tokenizer

        cls.max_model_len = config.actor_rollout_ref.rollout.max_model_len

        cls.rollout_n = config.actor_rollout_ref.rollout.n
        assert cls.rollout_n > 1

        cls.expand_branch_chain = config.actor_rollout_ref.rollout.spo.expand_branch_chain
        cls.expand_only_on_low_quality = config.actor_rollout_ref.rollout.spo.expand_only_on_low_quality
        cls.low_quality_trajectory_reward_threshold = config.actor_rollout_ref.rollout.spo.low_quality_trajectory_reward_threshold
        cls.branches_per_point = config.actor_rollout_ref.rollout.spo.branches_per_point
        cls.pick_branch_chain_root_method = config.actor_rollout_ref.rollout.spo.pick_branch_chain_root_method
        cls.n_branch_points = config.actor_rollout_ref.rollout.spo.n_branch_points
        assert cls.pick_branch_chain_root_method in ["random", "utility"], f"Unknown pick_branch_chain_root_method: {cls.pick_branch_chain_root_method}"
        if cls.pick_branch_chain_root_method == "random":
            cls.branching_strategy = RandomBranchingStrategy(max_model_len=cls.max_model_len)
        elif cls.pick_branch_chain_root_method == "utility":
            cls.branching_strategy = UtilitySamplingStrategy(max_model_len=cls.max_model_len,
                                                             prob_model_type=config.actor_rollout_ref.rollout.spo.utility_sampling.prob_model_type,
                                                             window_size=config.actor_rollout_ref.rollout.spo.utility_sampling.window_size,
                                                             position_bias=config.actor_rollout_ref.rollout.spo.utility_sampling.position_bias)


    async def _score_node(self, node: Node):
        result = await score_node(node=node, tokenizer=self.tokenizer)
        node.reward = result.reward
        node.reward_info = result
    
    @abstractmethod
    async def _step(self, parent_node: Node, sampling_params: Dict[str, Any], request_id: str) -> Node:
        raise NotImplementedError
        
    async def _rollout(self, node: Node, sampling_params, request_id: str) -> Node:
        current_node = node
        generated_chain: List[Node] = []
        while True:
            new_node = await self._step(current_node, sampling_params, request_id)
            generated_chain.append(new_node)
            if new_node.finish_reason == FINISH_REASON.COMPLETED or new_node.finish_reason == FINISH_REASON.EXCEED_LENGTH:
                break
            current_node = new_node
        assert len(generated_chain) > 0
        last_node = generated_chain[-1]
        await self._score_node(last_node)
        for i in range(len(generated_chain) - 1):
            generated_chain[i].children = [generated_chain[i + 1]]
        return generated_chain[0]

    async def _expand(self, node: Node, sampling_params: Dict[str, Any], rollout_n: int):
        assert node.finish_reason is None or node.finish_reason == FINISH_REASON.STOP
        tasks = []
        for rollout_idx in range(rollout_n):
            request_id = f"{node.node_id}:{rollout_idx}"
            tasks.append(asyncio.create_task(self._rollout(node, sampling_params, request_id)))
        new_children = await asyncio.gather(*tasks)
        if node.children is not None:
            node.children.extend(new_children)
        else:
            node.children = new_children

    def _get_chain_from_start_node(self, start_node: Node) -> List[Node]:
        chain = []
        current_node = start_node
        while current_node:
            chain.append(current_node)
            if current_node.children and len(current_node.children) > 0:
                current_node = current_node.children[0]
            else:
                break
        return chain

    def _node_to_val_output(self, node: Node, metrics: Dict[str, Any]) -> TSValAgentLoopOutput:
        assert len(node.prompt_ids) > 0
        assert len(node.response_ids) > 0
        assert len(node.response_ids) == len(node.response_mask)
        assert node.num_turns is not None
        assert node.reward is not None
        assert node.reward_info is not None

        return TSValAgentLoopOutput(
            prompt_ids=node.prompt_ids,
            response_ids=node.response_ids,
            response_mask=node.response_mask,
            num_turns=node.num_turns,
            reward=node.reward,
            reward_info=node.reward_info,
            metrics=metrics
        )

    def _node_to_train_output(self, node: Node, tree_id: str, metrics: Dict[str, Any]) -> TSTrainAgentLoopOutput:
        val_output = self._node_to_val_output(node, metrics)

        assert node.advantage is not None

        return TSTrainAgentLoopOutput(
            **val_output.model_dump(),
            tree_id=tree_id,
            node_id=node.node_id,
            advantage=node.advantage,
        )

    def _compress_chain(self, chain: List[Node]) -> Node:
        head = chain[0]
        last = chain[-1]

        prompt_ids = head.prompt_ids
        data_instance = head.data_instance

        finish_reason = last.finish_reason
        reward = last.reward
        reward_info = last.reward_info
        num_turns = last.num_turns

        response_ids = []
        response_mask = []

        for node in chain:
            response_ids.extend(node.response_ids)
            response_mask.extend(node.response_mask)

        return Node(node_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    data_instance=data_instance,
                    finish_reason=finish_reason,
                    num_turns=num_turns,
                    reward=reward,
                    reward_info=reward_info
                )
    
    async def _try_teacher_selection(self, chain: List[Node]) -> Optional[Node]:
        prompt_ids = chain[0].prompt_ids
        all_response_ids = [node.response_ids for node in chain]
        
        instruction, segments = await asyncio.gather(
            self.loop.run_in_executor(None, lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=True)),
            self.loop.run_in_executor(None, lambda: self.tokenizer.batch_decode(all_response_ids, skip_special_tokens=True))
        )
        
        steps_text = "\n\n".join([f"Step {i}:\n{seg}" for i, seg in enumerate(segments)])
        prompt = TEACHER_SELECTION_PROMPT_TEMPLATE.format(
            instruction=instruction, 
            reference=chain[0].data_instance["extra_info"]["answer"], # TODO: here assert reference answer is in [extra_info][answer]
            steps=steps_text
        )
        
        idx, reply = await call_teacher_with_retry(
            message=prompt,
            parse_fn=_parse_teacher_selection_reply,
            temperature_schedule=(0.0,), # only try once
            log_prefix="LLM Judge (teacher_selection)",
        )

        if idx is None:
            logger.warning(
                "Teacher selection did not return a index. "
                f"Raw reply: {reply}"
            )
            return None

        if idx > 0 and idx < len(chain):
            node = chain[idx - 1]
            if node.finish_reason == FINISH_REASON.STOP and \
                (len(node.prompt_ids) + len(node.response_ids) < self.max_model_len):
                return node
        
        return None

    async def _create_branching_selection(self, chain: List[Node], n_points: int) -> List[BranchingSelection]:
        target_nodes = self.branching_strategy.select_node(chain, n_points)
        selections = []
        for target_node in target_nodes:
            branch_chain_root_index = chain.index(target_node)

            target_node = Node(
                node_id=uuid4().hex,
                prompt_ids=target_node.prompt_ids,
                response_ids=target_node.response_ids,
                response_mask=target_node.response_mask,
                data_instance=target_node.data_instance,
                finish_reason=target_node.finish_reason,
                num_turns=target_node.num_turns
            )            

            selections.append(BranchingSelection(
                branch_chain_root=target_node,
                branch_chain_root_index=branch_chain_root_index,
                total_length=len(chain)
            ))

        return selections
    

    async def _process_one_branch_selection(
        self, 
        selection: BranchingSelection, 
        sampling_params: Dict[str, Any]
    ) -> Tuple[List[Node], BranchingFeedback, Optional[HardFailureSeed]]:
        branch_chain_root = selection.branch_chain_root
        await self._expand(branch_chain_root, sampling_params, rollout_n=self.branches_per_point)

        branch_nodes = []
        for child in branch_chain_root.children:
            branch_nodes.append(self._compress_chain(self._get_chain_from_start_node(child)))

        any_branch_success = any(node.reward is not None and node.reward > self.low_quality_trajectory_reward_threshold for node in branch_nodes)

        feedback = BranchingFeedback(
            total_length=selection.total_length,
            branch_chain_root_index=selection.branch_chain_root_index,
            is_success=any_branch_success
        )

        hard_failure_seed = None
        if not any_branch_success:
            hard_failure_seed = HardFailureSeed(
                prompt_ids=branch_chain_root.prompt_ids,
                response_ids=branch_chain_root.response_ids,
                data_instance=branch_chain_root.data_instance
            )

        return branch_nodes, feedback, hard_failure_seed

    async def _process_single_chain(self, 
                                    prompt_ids: List[int], 
                                    sampling_params: Dict[str, Any], 
                                    data_instance: Dict[str, Any]) -> Tuple[Node, List[List[Node]], List[HardFailureSeed], Dict[str, float]]:
        timing_metrics = {}
        root = Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=[],
            response_mask=[],
            data_instance=data_instance,
            num_turns=0
        )

        with simple_timer("generate_main_chain", timing_metrics):
            await self._expand(root, sampling_params, rollout_n=1)
        
        main_chain = [root] + self._get_chain_from_start_node(root.children[0])
        
        main_chain_node = self._compress_chain(main_chain)

        with simple_timer("generate_branch_chain", timing_metrics):
            branch_chain_node_groups: List[List[Node]] = []
            hard_failure_seeds_per_chain: List[HardFailureSeed] = []

            do_branch = False

            is_low_quality = (
                main_chain[-1].reward is not None and
                main_chain[-1].reward <= self.low_quality_trajectory_reward_threshold
            )

            do_branch = (
                self.expand_branch_chain and
                (not self.expand_only_on_low_quality or is_low_quality)
            )

            if do_branch:
                branching_selections = await self._create_branching_selection(main_chain, n_points=self.n_branch_points)

                tasks = []
                for selection in branching_selections:
                    task = asyncio.create_task(
                        self._process_one_branch_selection(selection, sampling_params)
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks)

                for branch_nodes, feedback, hard_failure_seed in results:
                    branch_chain_node_groups.append(branch_nodes)
                    
                    self.branching_strategy.update(feedback)
                    
                    if hard_failure_seed:
                        hard_failure_seeds_per_chain.append(hard_failure_seed)

        return main_chain_node, branch_chain_node_groups, hard_failure_seeds_per_chain, timing_metrics

    def _collect_train_outputs(
        self, 
        nodes: List[Node], 
        tree_id: str, 
        timing_metrics: Optional[List[Dict[str, Any]]] = None
    ) -> List[TSTrainAgentLoopOutput]:
        rewards = [n.reward for n in nodes]
        baseline = np.mean(rewards)
        
        outputs = []
        for i, node in enumerate(nodes):
            node.advantage = float(node.reward) - float(baseline)
            if abs(float(node.advantage)) < 1e-8:
                continue
            if timing_metrics is not None:
                metrics = timing_metrics[i]
            else:
                metrics = {}
            outputs.append(self._node_to_train_output(node, tree_id=tree_id, metrics=metrics))
        
        return outputs

    @rollout_trace_op
    async def _run_train(self, 
                        messages: List[Dict[str, Any]], 
                        sampling_params: Dict[str, Any], 
                        data_instance: Dict[str, Any]) -> Tuple[List[TSTrainAgentLoopOutput], List[TSTrainAgentLoopOutput], List[Dict]]:
        tree_id = data_instance["uid"]

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=getattr(self, 'add_generation_prompt', True), tokenize=True
            ),
        )

        tasks = []
        for _ in range(self.rollout_n):
            task = asyncio.create_task(
                self._process_single_chain(
                    prompt_ids=prompt_ids, 
                    sampling_params=sampling_params,
                    data_instance=data_instance
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        main_chain_nodes = []
        branch_chain_outputs = []
        hard_failure_seeds = []
        main_chain_timing_metrics = []

        for main_chain_node, branch_chain_node_groups, hard_failure_seeds_per_chain, timing_metrics in results:
            main_chain_nodes.append(main_chain_node)
            main_chain_timing_metrics.append(timing_metrics)
            
            for branch_chain_nodes in branch_chain_node_groups:
                branch_chain_outputs.extend(self._collect_train_outputs(
                    nodes=branch_chain_nodes, 
                    tree_id=tree_id
                ))

            for hard_failure_seed in hard_failure_seeds_per_chain:
                hard_failure_seeds.append(hard_failure_seed.to_dict())

        main_chain_outputs = self._collect_train_outputs(
            nodes=main_chain_nodes,
            tree_id=tree_id,
            timing_metrics=main_chain_timing_metrics
        )

        return main_chain_outputs, branch_chain_outputs, hard_failure_seeds
    
    @rollout_trace_op
    async def _run_val(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> TSValAgentLoopOutput:
        metrics = {}

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=getattr(self, 'add_generation_prompt', True), tokenize=True
            ),
        )

        root = Node(
            node_id=uuid4().hex,
            prompt_ids=prompt_ids,
            response_ids=[],
            response_mask=[],
            data_instance=data_instance,
            num_turns=0
        )

        with simple_timer("generate_main_chain", metrics):
            await self._expand(root, sampling_params, rollout_n=1)

        node = self._compress_chain(self._get_chain_from_start_node(root.children[0]))

        return self._node_to_val_output(node, metrics)
    
    async def run(
        self,
        mode: str,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        data_instance: Dict[str, Any],
    ) -> Union[TSValAgentLoopOutput, Tuple[List[TSTrainAgentLoopOutput], List[TSTrainAgentLoopOutput], List[Dict]]]:
        if mode == "train":
            return await self._run_train(messages, sampling_params, data_instance)
        elif mode == "validate":
            return await self._run_val(messages, sampling_params, data_instance)
        else:
            raise ValueError(f"Unknown mode: {mode}")