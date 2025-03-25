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

def sum (num1:int,num2:int)->int:
    """
    This function is used to add two numbers and return their sum.
    It takes two integers as inputs and returns an integer as output.
    """
    return  num1 + num2

def subtraction (num1:int,num2:int)->int:
    return num1 - num2

def multiplication (num1:int,num2:int)->int:
    """
    This function is used to multiply two numbers and return their product.
    It takes two integers as inputs and returns an integer as ouput.
    """
    return num1 * num2

#create function tools
sum_tool = FunctionTool.from_defaults(sum)
subtraction_tool = FunctionTool.from_defaults(subtraction)
multiplication_tool = FunctionTool.from_defaults(multiplication)


#define ReAct agent

react_agent = ReActAgent.from_tools(
            tools = [sum_tool,subtraction_tool,multiplication_tool],
            llm=llm,
            verbose=True #Set verbose for detailed logs
            )

response = react_agent.query("what is (10+45) multiply by 5 minus 75 ? use the tools provided")

print(response)


#******************************************************************
# SAMPLE RESPONSE : query - what is (10+45) multiply by 5 ?
#******************************************************************
#> Running step 2f5b343f-6270-4d63-977a-c6bcccb450fa. Step input: what is (10+45) multiply by 5 ? use the tools provided
#Thought: The current language of the user is: English. I need to use a tool to help me answer the question. First, I need to calculate the sum of 10 and 45.
#Action: sum
#Action Input: {'num1': 10, 'num2': 45}
#Observation: 55
#> Running step 2261f701-600d-42a6-87b5-b9e31c41bb6a. Step input: None
#Thought: Now that I have the sum of 10 and 45, which is 55, I can proceed to multiply it by 5. I will use the multiplication tool for this.
#Action: multiplication
#Action Input: {'num1': 55, 'num2': 5}
#Observation: 275
#> Running step 6ee87e4d-647b-41bc-a87a-b45e532c19b6. Step input: None
#Thought: I have now calculated the result of (10+45) multiplied by 5, which is 275. I can answer the question without using any more tools.
#Answer: 275
#275