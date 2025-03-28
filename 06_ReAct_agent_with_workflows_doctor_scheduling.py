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
from typing import Any, List,Union

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
from llama_index.utils.workflow import draw_all_possible_flows

#Preparation event for the request
class PrepEvent(Event):
    pass

#Input to the LLM
class InputEvent(Event):
    input : list[ChatMessage]

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
    async def new_user_msg(self, ctx:Context, ev:StartEvent) -> PrepEvent:
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
    async def prepare_chat_history(self, ctx:Context, ev:PrepEvent)-> InputEvent:
        # get chat history & format input
        chat_history = self.memory.get()
        current_reasoning = await ctx.get("current_reasoning",default= [])
        llm_input = self.formatter.format(
            self.tools, chat_history,
            current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(self, ctx:Context, ev:InputEvent) -> Union[ToolCallEvent,StopEvent]:
        chat_history = ev.input

        #Send prompt to LLM and get response
        response = await self.llm.achat(chat_history)

        #Analyze the response from LLM
        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            (await ctx.get("current_reasoning",default =[])).append(
                reasoning_step
            )
            print("*** LLM Returned : " ,reasoning_step)

            #If LLM returns ReACt is done
            if reasoning_step.is_done:
                self.memory.put( 
                        ChatMessage( role ="assistant", content=reasoning_step.response) 
                            )
                return StopEvent(
                    result ={
                        "response": reasoning_step.response,
                        "sources":[*self.sources],
                        "reasoning": await ctx.get("current_reasoning",default =[])
                    }
                ) 
            # If LLM says that tool calling is needed.
            elif isinstance(reasoning_step,ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                    ToolSelection(
                        tool_id = "fake",
                        tool_name = tool_name,
                        tool_kwargs = tool_args
                    )
                    ]     
                )
                
        except Exception as e:
            (await ctx.get("current_reasoning", default=[])).append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
        # if no tool calls or final response, iterate again
        return PrepEvent()
    
    @step
    async def handle_tool_calls(self, ctx : Context, ev : ToolCallEvent) -> PrepEvent:
        tools_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        # call tools There may be multiple tool calls in a single step
        print("*** Calling tools : ", tools_calls)
        for tool_call in tools_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue

            try:
                # call the tool
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                (await ctx.get("current_reasoning",default=[])).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )    
        # prep the next iteration.
        return PrepEvent()

#draw_all_possible_flows(SchedulingAgent, filename="scheduling_agent_flow.html")

# Create and execute the Scheduling Agent

#List of tools
scheduling_tools=[doctor_tool, schedule_appointment_tool]

context = """
You are a doctor scheduling assistant. You allow patients to search
for list of doctors by speciality. When a patient requests an appointement
with a specific doctor, you will also email the doctor with an appointment 
request.

Use doctors tool to search for doctor by disease / specialization, get 
their information including email ID.

To setup an appointment, use the schedule_appointment function passing in the 
required parameters captured from user input.

Use only the tools provided to answer questions and NOT your own memory.

"""

# Create the workflow agent
scheduling_agent = SchedulingAgent(
    llm = Settings.llm,
    tools = scheduling_tools,
    extra_context = context,
    timeout = 120,
    verbose = True
)

# wrapping your await statement within an async function and using asyncio.run() is required. 
# Python scripts don't automatically start an event loop, so you need asyncio.run() to create and manage the event loop for your asynchronous code.

#Example-1
#async def main():
#    response=await scheduling_agent.run(input="Which doctors are cardiologists?")
#    print("*******\n Response : ",response.get("response"))

#asyncio.run(main())


#*****************************************************
# ACTUAL RESPONSE FROM AGENT
#*****************************************************
# Running step new_user_msg
# Step new_user_msg produced event PrepEvent
# Running step prepare_chat_history
# Step prepare_chat_history produced event InputEvent
# Running step handle_llm_input
# *** LLM Returned :  thought='The current language of the user is: English. I need to use a tool to help me answer the question.' action='query_engine_tool' action_input={'input': 'cardiologists'}
# Step handle_llm_input produced event ToolCallEvent
# Running step handle_tool_calls
# *** Calling tools :  [ToolSelection(tool_id='fake', tool_name='query_engine_tool', tool_kwargs={'input': 'cardiologists'})]
# Step handle_tool_calls produced event PrepEvent
# Running step prepare_chat_history
# Step prepare_chat_history produced event InputEvent
# Running step handle_llm_input
# *** LLM Returned :  thought="I have the information about the cardiologist, but I don't have the name of the doctor. I need to use the query_engine_tool to get the list of doctors and find the cardiologist." action='query_engine_tool' action_input={'input': 'cardiologist'}
# Step handle_llm_input produced event ToolCallEvent
# Running step handle_tool_calls
# *** Calling tools :  [ToolSelection(tool_id='fake', tool_name='query_engine_tool', tool_kwargs={'input': 'cardiologist'})]
# Step handle_tool_calls produced event PrepEvent
# Running step prepare_chat_history
# Step prepare_chat_history produced event InputEvent
# Running step handle_llm_input
# *** LLM Returned :  thought='I can answer without using any more tools. I have the name of the cardiologist.' response='Dr. John Smith is a cardiologist with 15 years 
# of experience.' is_streaming=False
# Step handle_llm_input produced event StopEvent
# *******
#  Response :  Dr. John Smith is a cardiologist with 15 years of experience.

#Example-2

async def main():
    response=await scheduling_agent.run(input="Please setup an appointment with John Smith for Ben Jones next week in the afternoons")
    print("*******\n Response : ",response.get("response"))

asyncio.run(main())

#****************************
# RESPONSE
#****************************
# Running step new_user_msg
# Step new_user_msg produced event PrepEvent
# Running step prepare_chat_history
# Step prepare_chat_history produced event InputEvent
# Running step handle_llm_input
# *** LLM Returned :  thought="The current language of the user is: English. I need to use the query_engine_tool to find a doctor that matches the user's request, then use the schedule_appointment tool to setup the appointment." action='query_engine_tool' action_input={'input': 'doctors available next week in the afternoons'}
# Step handle_llm_input produced event ToolCallEvent
# Running step handle_tool_calls
# *** Calling tools :  [ToolSelection(tool_id='fake', tool_name='query_engine_tool', tool_kwargs={'input': 'doctors available next week in the afternoons'})]
# Step handle_tool_calls produced event PrepEvent
# Running step prepare_chat_history
# Step prepare_chat_history produced event InputEvent
# Running step handle_llm_input
# *** LLM Returned :  thought='The user needs to setup an appointment with John Smith for Ben Jones next week in the afternoons. Since the query_engine_tool did not provide the specific availability of the doctors, I will try to find the email of the doctor to setup the appointment.' action='query_engine_tool' action_input={'input': 'John Smith doctor email and specialty'}
# Step handle_llm_input produced event ToolCallEvent
# Running step handle_tool_calls
# *** Calling tools :  [ToolSelection(tool_id='fake', tool_name='query_engine_tool', tool_kwargs={'input': 'John Smith doctor email and specialty'})]
# Step handle_tool_calls produced event PrepEvent
# Running step prepare_chat_history
# Step prepare_chat_history produced event InputEvent
# Running step handle_llm_input
# *** LLM Returned :  thought='The current language of the user is: English. I need to use the query_engine_tool to find the email of Dr. John Smith to setup the appointment.' action='query_engine_tool' action_input={'input': 'Dr. John Smith email'}
# Step handle_llm_input produced event ToolCallEvent
# Running step handle_tool_calls
# *** Calling tools :  [ToolSelection(tool_id='fake', tool_name='query_engine_tool', tool_kwargs={'input': 'Dr. John Smith email'})]
# Step handle_tool_calls produced event PrepEvent
# Running step prepare_chat_history
# Step prepare_chat_history produced event InputEvent
# Running step handle_llm_input
# *** LLM Returned :  thought="The current language of the user is: English. I have the name of the doctor and the patient, and some information about the doctor's specialty and experience. I can now use the schedule_appointment tool to setup the appointment, assuming that the email will be handled internally by the system." action='schedule_appointment' action_input={'patient_name': 'Ben Jones', 'doctor_name': 'John Smith', 'scheduling_comments': 'next week in the afternoons'}
# Step handle_llm_input produced event ToolCallEvent
# Running step handle_tool_calls
# *** Calling tools :  [ToolSelection(tool_id='fake', tool_name='schedule_appointment', tool_kwargs={'patient_name': 'Ben Jones', 'doctor_name': 'John Smith', 'scheduling_comments': 'next week in the afternoons'})]
# Step handle_tool_calls produced event PrepEvent
# Running step prepare_chat_history
# Step prepare_chat_history produced event InputEvent
# Running step handle_llm_input
# *** LLM Returned :  thought="I can answer without using any more tools. I'll use the user's language to answer" response='The appointment with Dr. John Smith for Ben Jones has been successfully setup for next week in the afternoons.' is_streaming=False
# Step handle_llm_input produced event StopEvent
# *******
#  Response :  The appointment with Dr. John Smith for Ben Jones has been successfully setup for next week in the afternoons.