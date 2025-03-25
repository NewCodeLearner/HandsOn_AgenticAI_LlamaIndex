from dotenv import load_dotenv
load_dotenv()
import os
from llama_index.core.agent.react import ReActAgent
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.llms.groq import Groq



#define llm
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

llm = Groq(model = "llama-3.3-70b-versatile" , api_key = GROQ_API_KEY)


#create supporting functions

def sum(num1:int,num2:int)->int:
    return  num1 + num2

def subtraction (num1:int,num2:int)->int:
    return num1 - num2

def multiplication (num1:int,num2:int)->int:
    return num1 * num2