from typing import Any, Dict, List, Optional
from enum import Enum, auto

from dataclasses import dataclass, asdict

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


class FINISH_REASON(Enum):
    EXCEED_LENGTH = auto()
    STOP = auto()
    COMPLETED = auto()


@dataclass
class RewardInfo:
   reward: float
   completed: int
   finished: int = 0
   judgement_reply: Optional[str] = None


@dataclass
class Node:
  node_id: str
  prompt_ids: List[int]
  response_ids: List[int]
  response_mask: List[int]
  data_instance: Dict[str, Any]
  finish_reason: Optional[FINISH_REASON] = None
  num_turns: Optional[float] = None
  children: Optional[List["Node"]] = None
  reward: Optional[float] = None
  reward_info: Optional[RewardInfo] = None
  advantage: Optional[float] = None


class TSValAgentLoopOutput(AgentLoopOutput):
    reward: float
    reward_info: Optional[RewardInfo] = None


class TSTrainAgentLoopOutput(TSValAgentLoopOutput):
    tree_id: str
    node_id: str
    advantage: float


@dataclass
class HardFailureSeed:
    prompt_ids: List[int]
    response_ids: List[int]
    data_instance: Dict[str, Any]

    def to_dict(self):
        return asdict(self)