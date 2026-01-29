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
from typing import Any, Callable, Optional, Sequence, Tuple

import os
import logging
import random

import asyncio
from openai import AsyncOpenAI

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_MAX_CONCURRENCY = int(os.getenv("TEACHER_CONCURRENCY", "256"))

_TEACHER_MODEL_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENCY)

_TEACHER_MODEL_NAME = os.getenv("TEACHER_MODEL_NAME")
assert _TEACHER_MODEL_NAME is not None

_TEACHER_MODEL_CLIENT = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

async def _backoff_sleep(attempt: int, base: float = 0.5, cap: float = 8.0) -> None:
    # first attempt = 1 -> 0.5s to 0.7s
    # second attempt = 2 -> 1.0s to 1.2s
    # third attempt = 3 -> 2.0s to 2.2s
    
    delay = min(cap, base * (2 ** (attempt - 1)))
    delay += random.uniform(0, 0.2)
    await asyncio.sleep(delay)

async def call_teacher_with_retry(
    message: str,
    parse_fn: Callable[[str], Any],
    *,
    temperature_schedule: Sequence[float] = (0.0, 0.3, 0.7),
    max_attempts_per_temperature: int = 1,
    timeout: int = 1000,
    log_prefix: str = "LLM_JUDGE",
) -> Tuple[Optional[Any], str]:
    last_reply: str = ""
    global_attempt = 0

    for temperature in temperature_schedule:
        for _ in range(1, max_attempts_per_temperature + 1):
            global_attempt += 1
            try:
                async with _TEACHER_MODEL_SEMAPHORE:
                    response = await _TEACHER_MODEL_CLIENT.chat.completions.create(
                        model=_TEACHER_MODEL_NAME,
                        messages=[{"role": "user", "content": message}],
                        temperature=temperature,
                        timeout=timeout,
                    )
                reply = (response.choices[0].message.content or "").strip()
                last_reply = reply
            except Exception as e:
                logger.warning(
                    f"{log_prefix} request error "
                    f"(temperature={temperature}, attempt={global_attempt}): "
                    f"{type(e).__name__}: {e}"
                )
                await _backoff_sleep(global_attempt)
                continue

            try:
                parsed = parse_fn(reply)
                return parsed, reply
            except Exception as e:
                logger.warning(
                    f"{log_prefix} parse error\n"
                    f"(temperature={temperature}, attempt={global_attempt})\n"
                    f"Error: {e}\n"
                    f"Reply: {reply}"
                )
                await _backoff_sleep(global_attempt)
                continue

    return None, last_reply