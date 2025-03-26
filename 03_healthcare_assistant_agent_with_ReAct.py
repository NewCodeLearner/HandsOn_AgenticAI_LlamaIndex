from dotenv import load_dotenv
load_dotenv()
import os
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.json import JSONReader
from llama_index.llms.groq import Groq

# Set up LLM

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = Groq(model = "llama-3.3-70b-versatile" , api_key = GROQ_API_KEY)

# Set up wikipedia tools
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.tools.tool_spec.load_and_search import (
    LoadAndSearchToolSpec,
)
# Get list of wikipedia tools
wiki_spec = WikipediaToolSpec()
tool = wiki_spec.to_tool_list()[1]
wiki_tools = LoadAndSearchToolSpec.from_defaults(tool).to_tool_list()

#Review the list of tools
for tool in wiki_tools:
    print( "--------\n",tool.metadata.name, tool.metadata.description)
