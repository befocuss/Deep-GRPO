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
ACCURACY_EVALUATE_PROMPT = """
You are an impartial grader who provides clear reasoning for your decisions.

You will receive:
- a data science or math **Question**,
- a **Ground Truth Answer**, and
- a **Predicted Answer** produced by a model.

Your task is to decide whether the Predicted Answer is **correct** with respect to the Ground Truth Answer.

-----------------------
[QUESTION]
{question}

[GROUND TRUTH]
{ground_truth}

[PREDICTED ANSWER]
{predicted_answer}
-----------------------

Evaluation rules:
- The predicted answer must include a clear, explicit final answer.
- Pure calculations or partial steps without a final explicit answer are incorrect.
- Minor wording or formatting differences are allowed if the meaning is the same.
- If you are unsure whether they are equivalent, treat the answer as incorrect.

First, write your reasoning step by step, explaining:
1) What the ground truth answer means.
2) What the predicted answer means.
3) Whether they match exactly in content and conditions.

Then, at the very end, output EXACTLY ONE JSON code block with the following format:

```json
{{
  "correct": true or false
}}
```

Do not output any other JSON objects after this block.
"""