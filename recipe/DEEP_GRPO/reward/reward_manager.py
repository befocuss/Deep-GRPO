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
from typing import List

import asyncio

from recipe.DEEP_GRPO.reward import dabench
from recipe.DEEP_GRPO.reward import data_analysis
from recipe.DEEP_GRPO.reward import dsbench_data_analysis
from recipe.DEEP_GRPO.reward import dsbench_data_modeling
from recipe.DEEP_GRPO.reward import gsm8k
from recipe.DEEP_GRPO.reward import math
from recipe.DEEP_GRPO.reward import math_original
from recipe.DEEP_GRPO.reward import aime24
from recipe.DEEP_GRPO.reward import amc
from recipe.DEEP_GRPO.reward import minerva
from recipe.DEEP_GRPO.reward import olympiadbench
from recipe.DEEP_GRPO.reward import deepscaler
from recipe.DEEP_GRPO.reward import ds1000
from recipe.DEEP_GRPO.reward import wikitq
from recipe.DEEP_GRPO.reward import hybridqa
from recipe.DEEP_GRPO.reward import multihiertt
from recipe.DEEP_GRPO.reward import ottqa
from recipe.DEEP_GRPO.reward import finqa
from recipe.DEEP_GRPO.reward import project-data
from recipe.DEEP_GRPO.reward import dabstep_research
from recipe.DEEP_GRPO.reward import dabstep
from recipe.DEEP_GRPO.reward import qa_em_format
from recipe.DEEP_GRPO.protocol import RewardInfo, FINISH_REASON, Node


async def score_node(node: Node, tokenizer) -> RewardInfo:
    if node.finish_reason != FINISH_REASON.COMPLETED: # first check whether the response is complete, if not, assign 0 reward
        return RewardInfo(reward=0.0, completed=0, finished=0)

    loop = asyncio.get_running_loop()

    chat_history_str = await loop.run_in_executor(
            None, lambda: tokenizer.decode(node.prompt_ids + node.response_ids, skip_special_tokens=False)
    ) # include instruction, model response and special tokens

    data_instance = node.data_instance

    data_source = data_instance["data_source"]

    if data_source == "datatask":
        result = await data_analysis.compute_datatask_reward(instruction=data_instance["extra_info"]["instruction"],
                                                            chat_history_str=chat_history_str,
                                                            ground_truth=data_instance["reward_model"]["ground_truth"]
                                                            )
    
    elif data_source == "reasoning-table":
        result = await data_analysis.compute_tableqa_reward(instruction=data_instance["extra_info"]["instruction"],
                                                           chat_history_str=chat_history_str,
                                                           ground_truth=data_instance["reward_model"]["ground_truth"]
                                                          )
        
    elif data_source == "research":
        result = await data_analysis.compute_research_task_reward(instruction=data_instance["extra_info"]["instruction"],
                                                                 chat_history_str=chat_history_str,
                                                                )
        
    elif data_source == "DSBench_data_analysis":
        result = await dsbench_data_analysis.llm_as_judgement_accuracy(question=data_instance["extra_info"]["question"],
                                                                        chat_history_str=chat_history_str,
                                                                        ground_truth_answer=data_instance["reward_model"]["ground_truth"]
                                                                    )
    elif data_source == "DSBench_data_modeling":
        result = await dsbench_data_modeling.evaluate_submission(chat_history_str=chat_history_str,
                                                                eval_script_path=data_instance["extra_info"]["evaluation_script"],
                                                                submission_file=data_instance["reward_model"]["expected_file"],
                                                                ground_truth_file=data_instance["reward_model"]["ground_truth"],
                                                                baseline_result=data_instance["reward_model"]["baseline_result"],
                                                                gold_result=data_instance["reward_model"]["gold_result"])
    elif data_source == "DABench":
        result = dabench.compute_score(chat_history_str=chat_history_str,
                                       ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "GSM8K":
        result = gsm8k.compute_score(chat_history_str=chat_history_str,
                                     ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "DigitalLearningGmbH/MATH-lighteval":
        result = math.compute_score(chat_history_str=chat_history_str,
                                    ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "MATH":
        result = math.compute_score(chat_history_str=chat_history_str,
                                    ground_truth=data_instance["reward_model"]["ground_truth"])
    
    elif data_source == "AIME24":
        result = aime24.compute_score(chat_history_str=chat_history_str,
                                      ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "AMC":
        result = amc.compute_score(chat_history_str=chat_history_str,
                                   ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "Minerva":
        result = minerva.compute_score(chat_history_str=chat_history_str,
                                       ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "OlympiadBench":
        result = olympiadbench.compute_score(chat_history_str=chat_history_str,
                                             ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "DeepScaleR":
        result = deepscaler.compute_score(chat_history_str=chat_history_str,
                                          ground_truth=data_instance["reward_model"]["ground_truth"])
        
    elif data_source == "ds1000":
        result = await ds1000.evaluate_ds1000(chat_history_str=chat_history_str, code_context=data_instance["extra_info"]["code_context"])

    elif data_source == "wikitq":
        result = wikitq.evaluate_answer(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "hybridqa":
        result = hybridqa.reward_hybridqa_exact_match(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "multihiertt":
        result = multihiertt.reward_multihiertt_exact_match(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "ottqa":
        result = ottqa.reward_ottqa_exact_match(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])
    
    elif data_source == "finqa":
        result = finqa.reward_finqa(chat_history_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])

    elif data_source == "hotpotqa" or data_source == "2wikimultihopqa" or data_source == "musique" or data_source == "bamboogle":
        result = qa_em_format.compute_score_em(solution_str=chat_history_str, ground_truth=data_instance["reward_model"]["ground_truth"])
    
    elif data_source == "dabstep_research":
        result = await dabstep_research.llm_as_judgement_dabstep_report_quality(instruction=data_instance["extra_info"]["instruction"],
                                                                   chat_history_str=chat_history_str,
                                                                   checklist=data_instance["reward_model"]["ground_truth"]
                                                                )
    
    elif data_source == "dabstep":
        result = await dabstep.llm_as_judgement_dabstep_dummy(instruction=data_instance["extra_info"]["instruction"],
                                                              chat_history_str=chat_history_str,
                                                              ground_truth=data_instance["reward_model"]["ground_truth"]
                                                            )
    # elif data_source == "project-data_compare":
    #     result = await project-data.llm_as_judgement_accuracy(model_solution=text,
    #                                                    ground_truth_answer=data_instance["reward_model"]["ground_truth"],
    #                                                    query=data_instance["extra_info"]["query"],
    #                                                    candidate_algorithms=data_instance["extra_info"]["candidate_algorithms"]
    #                                                 )

    else:
        raise ValueError(f"Unknown data source {data_source}")
    
    result.finished = 1 # we have checked whether the response is complete at the beginning
    
    return result