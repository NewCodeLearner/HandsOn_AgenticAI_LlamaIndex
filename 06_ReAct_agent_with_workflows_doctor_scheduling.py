from dotenv import load_dotenv
load_dotenv()
import os
from llama_index.core import VectorStoreIndex,Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.json import JSONReader
from llama_index.llms.groq import Groq
from llama_index.core.agent.react.base import ReActAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import nest_asyncio ,asyncio

#Used by LlamaIndex
nest_asyncio.apply()

# Setup LLM and Embedding
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
Settings.llm = Groq(model = "llama-3.3-70b-versatile" , api_key = GROQ_API_KEY)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# Setup Doctor database tool
from llama_index.readers.json import JSONReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import  VectorStoreIndex
from llama_index.core.tools import QueryEngineTool

doctors_doc = JSONReader().load_data(input_file='data/Doctors database.json')

splitter = SentenceSplitter(chunk_size=200)
doc_nodes = splitter.get_nodes_from_documents(doctors_doc)
#Index the document in memory
doctor_index = VectorStoreIndex(doc_nodes)

doctor_query_engine = doctor_index.as_query_engine()

doctor_tool = QueryEngineTool.from_defaults(
    query_engine = doctor_query_engine,
    description=(
        """Provides the list of doctors, 
            diseases they specialize in (or speciality),
            years of experience and 
            their contact email Id to setup appointments"""
    )
    )

# Setup scheduling tool


# Setup custom events


# Create the custom ReAct workflow





# draw_all_possible_flows(SchedulingAgent, filename="scheduling_agent_flow.html")


# Create and execute the Scheduling Agent

