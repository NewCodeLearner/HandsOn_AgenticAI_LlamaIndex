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
import datetime
import csv

def schedule_appointment(patient_name:str, 
                         doctor_name:str, 
                         scheduling_comments:str) -> bool :
    """
    This function is used to schedule a doctor appointment. It takes 3 inputs
    patient_name : The name of the patient for who the appointment is required.
    doctor_name : The name of the doctor to setup the appointment with.
    scheduling_comments: Additional information on requested date, time etc. for the appointment

    The function returns True if the appointment setup is successful. False otherwise.
    """

    #Capture current date and month
    requested_date = datetime.datetime.now().strftime("%b-%d")

    #Capture appointment details
    appointment_details = [requested_date,
                           patient_name,
                           doctor_name,
                           scheduling_comments
                          ]

    with open("data/Doctor appointment requests.csv","a",newline="") as appts:
        writer = csv.writer(appts)
        writer.writerow(appointment_details)
        return True
    
    #if otherwise.
    return False

#test code
#print(schedule_appointment("Tim Jones","Jack Smith","Monday afternoons"))

#Create a function tool for appointments
from llama_index.core.tools import FunctionTool
schedule_appointment_tool = FunctionTool.from_defaults(fn=schedule_appointment)


# Setup custom events
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from typing import Any, List

from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

#Preparation event for the request
class PrepEvent(Event):
    pass

#Input to the LLM
class InputEvent(Event):
    Input = list[ChatMessage]

#Trigger tool calls
class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


model =Settings.llm
# Create the custom ReAct workflow
class SchedulingAgent(Workflow):
    def __init__(
            self,
            *args:Any,
            llm: LLM ,
            tools:list[BaseTool],
            extra_context:str,
            **kwargs:Any,

    )-> None:
        #call the parent class init
        super().__init__(*args,**kwargs)

        #copy input to instance varialbes
        self.tools = tools
        self.llm = Settings.llm
        #setup memory to track request
        self.memory = ChatMemoryBuffer.from_defaults(llm=Settings.llm)
        self.formatter = ReActChatFormatter(context=extra_context or "")
        self.output_parser = ReActOutputParser()
        self.sources =[]
    
    @step
    async def new_user_msg(self,ctx:Context, ev: StartEvent) -> PrepEvent:
        #clear sources
        self.sources =[]

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role='user',content=user_input)
        self.memory.put(user_msg)

        # clear current reasoning since its a new request
        await ctx.set("current_reasoning",[])

        return PrepEvent()
    
    @step
    async def prepare_chat_history(self,ctx:Context,ev:PrepEvent)-> InputEvent:
        # get chat history & format input
        chat_history = self.memory.get()
        current_reasoning = await ctx.get("current_reasoning",default= [])
        llm_input = self.formatter.format(
            self.tools, chat_history,
            current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)


# draw_all_possible_flows(SchedulingAgent, filename="scheduling_agent_flow.html")


# Create and execute the Scheduling Agent

