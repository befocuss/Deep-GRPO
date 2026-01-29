import re
import logging
import os
import json

from openai import AsyncOpenAI

from recipe.spo.protocol import RewardInfo
from recipe.spo.utils import _TEACHER_MODEL_SEMAPHORE


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


SYSTEM_PROMPT = """You are a strict evaluation judge for a reinforcement learning setup.

Your job:

1. Read:
   - A business question in Chinese (`user_query`),
   - A list of `candidate_algorithms` (internal hint for what algorithms are relevant),
   - A natural-language evaluation checklist (`judge_rubric`),
   - A Student's answer (`student_answer`).

2. Treat `judge_rubric` as a HARD CHECKLIST:
   - Go through each item in the checklist and decide whether the Student’s answer satisfies it.
   - Be strict: if an item is only partially or weakly satisfied, treat it as NOT satisfied.

3. After careful evaluation, you must produce TWO parts in order:
   - First: a detailed reasoning section in English, where you explicitly reference the checklist items and explain which are satisfied or not.
   - Second: a JSON object summarizing your judgment, inside a Markdown ```json code block.

JSON schema:

- `reward`: 1 if and ONLY if the Student’s answer satisfies ALL important checklist items in `judge_rubric`; otherwise 0.
- `passed_items`: a list of short English descriptions of checklist requirements that were clearly satisfied.
- `failed_items`: a list of short English descriptions of checklist requirements that were NOT satisfied (or only weakly satisfied).
- `analysis`: a short English paragraph summarizing your overall judgment.

Important rules:

- Be conservative: if you are unsure whether a checklist item is satisfied, treat it as NOT satisfied.

Output format requirements:

1. First output a section starting with the header line:
   `### Reasoning`
   and then write your detailed reasoning in English.

2. Then, on a new line, output:
   `### Final JSON`

3. Immediately after that, output the JSON object INSIDE a Markdown code block with language tag `json`, like:

   ```json
   {
     "reward": 0,
     "passed_items": [],
     "failed_items": [],
     "analysis": "..."
   }"""


USER_PROMPT_TEMPLATE = """You are given the following information for evaluation:

[Business Question in Chinese]
{user_query}

[Candidate Algorithms (for your reference)]
{candidate_algorithms_as_text}

[Evaluation Checklist (Judge Rubric)]
{judge_rubric}

[Student Answer]
{student_answer}

Now, follow the checklist strictly and decide whether the Student’s answer deserves reward = 1 or 0.

- reward = 1  → ALL important checklist items in `judge_rubric` are clearly satisfied.
- reward = 0  → At least one important checklist item is NOT satisfied (or only weakly satisfied).

Remember to:

1. First write a detailed reasoning section under the header `### Reasoning`, explaining:
   - Which checklist items are satisfied and why,
   - Which checklist items are not satisfied and why,
   - How well the answer resolves the original business question,
   - Whether there is flaw in the student answer. Does the conclusion logically follow from the analysis?

2. Then output `### Final JSON` and a single JSON object inside a ```json code block, with the fields:
   - "reward" (0 or 1),
   - "passed_items" (list of strings),
   - "failed_items" (list of strings),
   - "analysis" (string)."""


def check_ciecc_import(student_answer: str):
    code_blocks = re.findall(r'<Code>(.*?)</Code>', student_answer, re.DOTALL)
    
    for block in code_blocks:
        if 'import ciecc' in block or 'from ciecc import' in block:
            return True
    return False

def check_for_result_tag(student_answer: str):
    if "<Result>" in student_answer:
        return True
    return False

async def llm_as_judgement_accuracy(model_solution: str, 
                                    ground_truth_answer: str,
                                    query: str,
                                    candidate_algorithms: list[str],
                                    client: AsyncOpenAI, 
                                    model: str) -> RewardInfo:

    import_reward = 0.5 if check_ciecc_import(student_answer=model_solution) else 0
    retrieve_reward = 0.5 if check_for_result_tag(student_answer=model_solution) else 0
    
    candidate_algorithms_as_text = "\n".join(f"- {alg}" for alg in candidate_algorithms)

    message = USER_PROMPT_TEMPLATE.format(
        user_query=query,
        student_answer=model_solution,
        judge_rubric=ground_truth_answer,
        candidate_algorithms_as_text=candidate_algorithms_as_text,
    )
    reply = ""
    try:
        async with _TEACHER_MODEL_SEMAPHORE:
            response = await client.chat.completions.create(
                model=model, messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ], timeout=1000, temperature=0
            )
        reply = response.choices[0].message.content.strip()

        reply_match = re.search(r"```(?:json)?(.*?)```", reply, re.DOTALL)
        score_str = reply_match.group(1).strip() if reply_match else reply
        score_dict = json.loads(score_str)

        score_reward = score_dict["reward"]
        return RewardInfo(reward=import_reward + retrieve_reward + float(score_reward),
                            completed=1,
                            judgement_reply=reply)

    except Exception as e:
        logger.warning(f"LLM Judge error: {type(e).__name__}: {e}")
        return RewardInfo(reward=import_reward + retrieve_reward,
                            completed=0,
                            judgement_reply=reply)

