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
TEACHER_SELECTION_PROMPT_TEMPLATE = """
You are an expert Reasoning and Error Analysis Agent.
I will provide you with a **Problem**, a **Reference Solution** (Standard Answer), and a **Student's Incorrect Trajectory**.

Your task is to compare the Student's Trajectory against the Reference Solution and identify the **FIRST step** where the student made a mistake.
The mistake could be a logical error, a calculation error, a wrong code implementation, or a hallucination.

### Input Data
**Problem**:
{instruction}

**Reference Solution**:
{reference}

**Student Trajectory**:
{steps}

### Instructions
1. **Analyze Step-by-Step**: 
   - Go through the student's steps one by one.
   - Explicitly state whether each step is correct or incorrect and why.
2. **Identify the First Error**:
   - Locate the exact index (0-based) of the first step that deviates from the correct logic.
3. **Output JSON**:
   - Finally, output the result in a JSON object containing the key "first_error_step_index".

### Output Format
First, provide your analysis. Then, output the JSON in the following format:

### Analysis
[Your step-by-step reasoning here...]

### Result
```json
{{
    "first_error_step_index": <int>
}}
```

(If the very first step is wrong, return 0. If no error is found, return -1.)

Now, start your analysis.
"""