import os
import asyncio
from concurrent.futures import ThreadPoolExecutor


DOMAIN_API_DOCS_BASE_DIR = os.environ["DOMAIN_API_DOCS_BASE_DIR"]

_IO_EXECUTOR = ThreadPoolExecutor(max_workers=128)


async def retrieve_api_doc(query: str) -> str:
    if not query:
        return "Error: Empty query provided."
    
    loop = asyncio.get_running_loop()

    
    filename = f"{query.strip()}.md"
    file_path = os.path.join(DOMAIN_API_DOCS_BASE_DIR, filename)

    if not os.path.exists(file_path):
        return f"Error: Documentation for algorithm '{query}' not found. Please check the algorithm name."

    try:
        content = await loop.run_in_executor(
            _IO_EXECUTOR, 
            lambda: open(file_path, "r", encoding="utf-8").read()
        )
        return content
    except Exception as e:
        return f"Error reading documentation for '{query}': {str(e)}"