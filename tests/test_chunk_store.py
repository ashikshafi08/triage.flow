import pytest
from src.chunk_store import InMemoryChunkStore

def test_chunk_roundtrip():
    store = InMemoryChunkStore()
    data = "A" * 20000
    cid = store.store(data)
    assert store.retrieve(cid) == data 