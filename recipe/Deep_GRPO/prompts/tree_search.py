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