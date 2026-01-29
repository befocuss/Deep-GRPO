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
import asyncio
import json
import os
import logging
import argparse
from typing import List, Dict, Optional
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel("INFO")


SYSTEM_PROMPT = """
You are an expert AI Data Synthesizer.
You are provided with a User Question and a "Student Trajectory Prefix".

Your task is to generate TWO distinct training trajectories (Bifurcation).
For each path, you must output the **Prefix Used** (for reference) and the **Full Response** (The complete text starting with that prefix).

### Trajectory 1: The Self-Reflection Path (Repair)
- **Goal**: Continue the student's current stream of thought (even if flawed), then trigger a reflection and fix it.
- **Construction**:
    1. **Prefix Used**: The **EXACT** Student Trajectory Prefix.
    2. **Full Response**: 
       - Start with the `Prefix Used`.
       - If the prefix cuts off, finish the word/sentence naturally.
       - Then, **IMMEDIATELY** trigger a reflection (e.g., "Wait...", "Actually, I misread...").
       - **Correction Phase**: 
         - **Narrative Fix**: Correct the mistake using a natural flow, NOT a structured list.
         - Example: Instead of saying "Let's correct this: 1. A=50...", say "Wait, Anna actually has 50, not 18. That means John has..."
         - **Overwrite Logic**: Directly update the incorrect values/logic in the stream of thought and move forward.
       - **CRITICAL CONSTRAINT**: 
         - **NO RESETTING**: Do not say "Let's start over" or "Let's recalculate everything".
         - **NO RE-LISTING**: Do not generate a new numbered list (e.g., "1... 2... 3...") to re-state facts that were already established, unless necessary for a complex multi-step logical deduction.
         - Keep the tone uniform with the prefix.

### Trajectory 2: The Optimal Path (Clean)
- **Goal**: Solve the problem efficiently from the last valid point.
- **Construction**:
    1. **Prefix Used**: 
       - CRITICAL STEP: Carefully review the Student Prefix step-by-step.
       - Identify the **EXACT point** where the logic first became incorrect or factually wrong.
       - Extract ONLY the text **BEFORE** that error.
       - If the very first sentence is wrong, `Prefix Used` should be empty (or just the introductory "Let's think step by step:").
    2. **Full Response**:
       - Start with the `Prefix Used`.
       - Continue directly with the **CORRECT** logic.
       - **Style Constraint**: Mimic the student's original tone, formatting, and reasoning depth.
       - Do NOT mention the error. Do NOT say "Wait...".

### Output Format (JSON):
{
  "repair_path": {
      "prefix_used": "string (The prefix text you started with)",
      "full_response": "string (The COMPLETE response: prefix + completion + reflection + local correction + continuation)"
  },
  "clean_path": {
      "prefix_used": "string (The valid substring you started with)",
      "full_response": "string (The COMPLETE response: valid_prefix + optimal solution)"
  }
}
"""

USER_TEMPLATE = """
### User Question:
{user_query}

### Student Trajectory Prefix:
{student_prefix}
"""


class DataSynthesizer:
    def __init__(self, api_key: str, base_url: str, model: str, concurrency: int):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)

    async def _call_teacher(self, user_query: str, student_prefix: str) -> Optional[Dict]:
        try:
            async with self.sem:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(
                            user_query=user_query, 
                            student_prefix=student_prefix
                        )}
                    ],
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"API Call failed: {e}")
            return None

    def _construct_samples(self, user_query: str, student_prefix: str, result: Dict) -> List[Dict]:
        samples = []
        
        # --- 1. Repair Path ---
        path_data = result.get("repair_path", {})
        prefix_used = path_data.get("prefix_used", "")
        full_response = path_data.get("full_response", "")
        
        if prefix_used and full_response:
            if full_response.startswith(prefix_used) and student_prefix == prefix_used:
                samples.append({
                    "source": "teacher_repair",
                    "prompt": user_query,
                    "response": full_response,
                    "prefix": prefix_used
                })

        # --- 2. Clean Path ---
        path_data = result.get("clean_path", {})
        prefix_used = path_data.get("prefix_used", "")
        full_response = path_data.get("full_response", "")
        
        if full_response:
            if full_response.startswith(prefix_used) and student_prefix.startswith(prefix_used):
                samples.append({
                    "source": "teacher_clean",
                    "prompt": user_query,
                    "response": full_response,
                    "prefix": prefix_used
                })
            
        return samples

    async def process_seed(self, line: str, idx: int) -> List[Dict]:
        try:
            data = json.loads(line)
            user_query = data.get("prompt_text", "")
            student_prefix = data.get("response_text", "")

            if not user_query or not student_prefix:
                return []

            result = await self._call_teacher(user_query, student_prefix)
            if not result:
                return []

            return self._construct_samples(user_query, student_prefix, result)

        except Exception as e:
            logger.error(f"Error line {idx}: {e}")
            return []

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    # parser.add_argument("--input_file", type=str, default="/home/user/data/checkpoints/Qwen2.5-0.5B-Instruct-GSM8K-CRPO-tp-pb2-b8-debug-01051500/hard_failure_seeds.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    # parser.add_argument("--output_file", type=str, default="/home/user/data/checkpoints/Qwen2.5-0.5B-Instruct-GSM8K-CRPO-tp-pb2-b8-debug-01051500/synthesized_data.jsonl")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base_url", type=str, default="https://api.ai.cs.ac.cn/v1")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    # parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error("Input file not found.")
        return

    synthesizer = DataSynthesizer(args.api_key, args.base_url, args.model, args.concurrency)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = [l for l in f if l.strip()]

    tasks = [synthesizer.process_seed(line, i) for i, line in enumerate(lines)]
    results = await tqdm_asyncio.gather(*tasks)
    flat_samples = [item for sublist in results for item in sublist]

    if not flat_samples:
        logger.error("No samples synthesized.")
        return

    df = pd.DataFrame(flat_samples)

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(args.output_file, index=False)
    print(f"Synthesis Complete. Saved {len(df)} samples to {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
