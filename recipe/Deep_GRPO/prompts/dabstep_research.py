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
PROMPT = """You are a data science evaluation assistant. Here's a generated data science report based on the user instruction. Your task is to comprehensively evaluate the quality of the generated data science report, based on the provided user instruction [INSTRUCTION], a checklist offering reference points for an ideal report [CHECKLIST], and the generated report [REPORT].

Evaluate across two dimensions (1–5 scale):

- **Content**: Relevance, comprehensiveness, and insightfulness.
- **Format**: Structure, readability, and professionalism.

### [INSTRUCTION]:
{instruction}

### [CHECKLIST]:
{checklist}

### [REPORT]:
{report}

Return your evaluation strictly as JSON:
```json
{{
"Content": <score>,
"Format": <score>
}}
```"""