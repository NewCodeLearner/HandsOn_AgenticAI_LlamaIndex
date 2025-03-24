from dotenv import load_dotenv
load_dotenv()
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.llms.groq import Groq
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


#Set up groq client for llm

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

Settings.llm = Groq(model = "llama-3.3-70b-versatile" , api_key = GROQ_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')


documents = SimpleDirectoryReader('data/').load_data()
print('documents created')
index = VectorStoreIndex.from_documents(documents)
print('index created')
query_engine = index.as_query_engine(llm=Settings.llm)
print('query_engine created')
response = query_engine.query("which foundational models are mentioned in the paper?")

print(response)