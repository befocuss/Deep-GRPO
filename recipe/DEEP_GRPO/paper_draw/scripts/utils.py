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
def biased_ema(data, smoothing_weight=0.99):
    smoothed_data = []
    last = 0
    num_accum = 0
    for next_val in data:
        num_accum += 1
        last = last * smoothing_weight + (1 - smoothing_weight) * next_val
        debias_weight = 1.0 - pow(smoothing_weight, num_accum)
        smoothed_data.append(last / debias_weight)
    return smoothed_data