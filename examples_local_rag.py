import os
from src.local_repo_loader import clone_repo_to_temp
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import faiss
import nest_asyncio
import openai
import sys

nest_asyncio.apply()

# Set your OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("ERROR: OPENAI_API_KEY environment variable is not set or empty.")
    print("Please set a valid OpenAI API key using 'export OPENAI_API_KEY=your_key_here'")
    sys.exit(1)

openai.api_key = openai_api_key

# Set embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_api_key)
Settings.embed_model = embed_model
embedding_dim = 1536  # 3072 for text-embedding-3-large

# Example repo and query
repo_url = "https://github.com/huggingface/trl.git"
branch = "main"
query = "How does the PPOTrainer work? Which files should I look at?"

with clone_repo_to_temp(repo_url, branch) as repo_path:
    print(f"Cloned repo to: {repo_path}")
    # Load documents from the local repo
    documents = SimpleDirectoryReader(repo_path).load_data()
    print(f"Loaded {len(documents)} documents from repo.")

    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Query
    query_engine = index.as_query_engine(similarity_top_k=10, verbose=True)
    response = query_engine.query(query)
    print("\nRAG Response:")
    print(response) 