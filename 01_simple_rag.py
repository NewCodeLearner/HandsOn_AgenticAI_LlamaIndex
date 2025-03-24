from dotenv import load_dotenv
load_dotenv()
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from groq  import Groq



#Set up groq client for llm

groq_client = Groq(api_key='api_key')