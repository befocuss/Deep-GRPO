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
from openai import OpenAI


client = OpenAI(
  api_key="my_key",
  base_url="http://placeholder-api-server:8000/v1"
)

messages = [{
  "role": "user",
  "content": "Hello"
}]

response = client.chat.completions.create(
                    model="Qwen3-235B-A22B-Instruct-2507-AWQ",
                    messages=messages,
                    timeout=30
                )

print(response)