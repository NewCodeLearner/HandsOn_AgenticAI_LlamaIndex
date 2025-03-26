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


# Set up LLM

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
Settings.llm = Groq(model = "llama-3.3-70b-versatile" , api_key = GROQ_API_KEY)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# Create introspection agent
from llama_index.agent.introspective import SelfReflectionAgentWorker
from llama_index.agent.introspective import IntrospectiveAgentWorker
from llama_index.core.llms import ChatMessage, MessageRole

#Setup the reflection agent worker
self_reflection_agent_worker = SelfReflectionAgentWorker.from_defaults(
    llm=Settings.llm,
    verbose=True,
)

#Setup the introspective agent worker
introspective_agent_worker = IntrospectiveAgentWorker.from_defaults(
    reflective_agent_worker=self_reflection_agent_worker,
    main_agent_worker=None,
    verbose=True
)
