from dotenv import load_dotenv
load_dotenv()
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.llms.groq import Groq
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

#Set up groq client for llm

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

Settings.llm = Groq(model = "llama-3.3-70b-versatile" , api_key = GROQ_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')


documents = SimpleDirectoryReader(input_files=['data/whitepaper.pdf']).load_data()
print('documents created')

text_splitter = SentenceSplitter(chunk_size =1024)
nodes = text_splitter.get_nodes_from_documents(documents,show_progress=True)

index = VectorStoreIndex.from_documents(documents,node_parser = nodes)
print('index created')

query_engine = index.as_query_engine(llm=Settings.llm)
print('query_engine created')

response = query_engine.query("which foundational models are mentioned in the paper?")

print(response)