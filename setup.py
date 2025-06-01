from setuptools import setup, find_packages

setup(
    name="triage-flow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "llama-index-core",
        "llama-index-vector-stores-faiss",
        "openai",
        "python-dotenv",
        "faiss-cpu",
        "gitpython",
        "nest-asyncio",
        "pydantic",
        "requests",
        "tqdm",
        "aiohttp",
        "pydantic-settings",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "python-dotenv",
        "pydantic",
        "llama-index-retrievers-bm25",
        "llama-index-postprocessor-cohere-rerank",
        
    ],
) 