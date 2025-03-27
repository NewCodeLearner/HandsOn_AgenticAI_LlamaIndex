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

#Create a system prompt defining the function of the agent
system_prompt="""
You are an Product specification summarizer who can summarize a product specification.
For the input provided, create a summary with less than 100 words.

Ensurethat the summary focuses on performance specifications
and safety features.
"""

chat_history =[
    ChatMessage(
        content = system_prompt,
        role = MessageRole.SYSTEM
    )
]

# create the agent
introspective_agent = introspective_agent_worker.as_agent(
    chat_history=chat_history,
    verbose=True
)

# Execute the introspection Agent

from llama_index.readers.file import PyMuPDFReader

#Input file for summarization
loader=PyMuPDFReader()
docs=loader.load(file_path="data/EcoSprint_Specification_Document.pdf")


#Pick the first page of the doc as content
source_content=docs[0].text
#print(source_content)


# wrapping your await statement within an async function and using asyncio.run() is required. 
# Python scripts don't automatically start an event loop, so you need asyncio.run() to create and manage the event loop for your asynchronous code.
async def main():
    response = await introspective_agent.achat(source_content)
    print("\n", response.response)

asyncio.run(main())


#********************************************************
# ACTUAL RESPONSE FROM AGENT
#********************************************************
#********************************************************
#  > Running step 91fcadaa-cc79-489a-a583-db22dbef5a9e. Step input: EcoSprint Specification Document
#********************************************************
#  1. Overview
#  ●The EcoSprint is a revolutionary electric vehicle (EV) designed for efficiency and
#  performance. With its sleek design and state-of-the-art technology, the EcoSprint
#  appeals to environmentally conscious drivers who don't want to compromise on style or
#  driving experience. Ideal for city driving and daily commutes, the EcoSprint offers a
#  perfect blend of comfort, sustainability, and innovation.
#  2. Design Specifications
#  ●Exterior Design: The EcoSprint boasts a modern and aerodynamic silhouette, featuring
#  smooth lines and a compact form factor. Available in colors like Midnight Black, Ocean
#  Blue, and Pearl White, it's a head-turner on the road.
#  ●Interior Design: Inside, the EcoSprint is a realm of comfort and luxury. It offers a
#  spacious cabin with seating for five, premium upholstery, and customizable ambient
#  lighting.
#  3. Performance Specifications
#  ●Engine and Motor: Powered by a high-efficiency electric motor, the EcoSprint delivers
#  200 horsepower and 300 Nm of torque, providing a smooth and responsive driving
#  experience.
#  ●Battery and Range: Equipped with a 50 kWh lithium-ion battery, it offers an impressive
#  range of up to 250 miles on a single charge. Charging is convenient with options for
#  home-charging setups and public charging stations.
#  ●Acceleration and Top Speed: The vehicle accelerates from 0 to 60 mph in just 7.3
#  seconds and has a top speed of 120 mph.
#  4. Technology and Features
#  ●Infotainment System: A state-of-the-art infotainment system with a 10-inch touchscreen,
#  voice control, and smartphone integration for both Android and iOS devices.
#  ●Driver Assistance Systems: Includes advanced features like adaptive cruise control,
#  lane-keeping assist, and automatic emergency braking.
#  ●Connectivity: Remote monitoring and control via a smartphone app, enabling
#  pre-conditioning of the vehicle, charging status checks, and more.
#  5. Safety and Security
#  ●Safety Features: High-rated safety features including multiple airbags, a reinforced
#  frame, and blind-spot monitoring.
#  ●Security Features: Advanced security systems with remote locking/unlocking, GPS
#  tracking, and intrusion alarms.
#  
#********************************************************
#  Added user message to memory: EcoSprint Specification Document
#********************************************************
#  1. Overview
#  ●The EcoSprint is a revolutionary electric vehicle (EV) designed for efficiency and
#  performance. With its sleek design and state-of-the-art technology, the EcoSprint
#  appeals to environmentally conscious drivers who don't want to compromise on style or
#  driving experience. Ideal for city driving and daily commutes, the EcoSprint offers a
#  perfect blend of comfort, sustainability, and innovation.
#  2. Design Specifications
#  ●Exterior Design: The EcoSprint boasts a modern and aerodynamic silhouette, featuring
#  smooth lines and a compact form factor. Available in colors like Midnight Black, Ocean
#  Blue, and Pearl White, it's a head-turner on the road.
#  ●Interior Design: Inside, the EcoSprint is a realm of comfort and luxury. It offers a
#  spacious cabin with seating for five, premium upholstery, and customizable ambient
#  lighting.
#  3. Performance Specifications
#  ●Engine and Motor: Powered by a high-efficiency electric motor, the EcoSprint delivers
#  200 horsepower and 300 Nm of torque, providing a smooth and responsive driving
#  experience.
#  ●Battery and Range: Equipped with a 50 kWh lithium-ion battery, it offers an impressive
#  range of up to 250 miles on a single charge. Charging is convenient with options for
#  home-charging setups and public charging stations.
#  ●Acceleration and Top Speed: The vehicle accelerates from 0 to 60 mph in just 7.3
#  seconds and has a top speed of 120 mph.
#  4. Technology and Features
#  ●Infotainment System: A state-of-the-art infotainment system with a 10-inch touchscreen,
#  voice control, and smartphone integration for both Android and iOS devices.
#  ●Driver Assistance Systems: Includes advanced features like adaptive cruise control,
#  lane-keeping assist, and automatic emergency braking.
#  ●Connectivity: Remote monitoring and control via a smartphone app, enabling
#  pre-conditioning of the vehicle, charging status checks, and more.
#  5. Safety and Security
#  ●Safety Features: High-rated safety features including multiple airbags, a reinforced
#  frame, and blind-spot monitoring.
#  ●Security Features: Advanced security systems with remote locking/unlocking, GPS
#  tracking, and intrusion alarms.
#********************************************************
#  > Running step fee3480d-8005-40bf-bff0-1776963c22c9. Step input: EcoSprint Specification Document
#********************************************************
#  1. Overview
#  ●The EcoSprint is a revolutionary electric vehicle (EV) designed for efficiency and
#  performance. With its sleek design and state-of-the-art technology, the EcoSprint
#  appeals to environmentally conscious drivers who don't want to compromise on style or
#  driving experience. Ideal for city driving and daily commutes, the EcoSprint offers a
#  perfect blend of comfort, sustainability, and innovation.
#  2. Design Specifications
#  ●Exterior Design: The EcoSprint boasts a modern and aerodynamic silhouette, featuring
#  smooth lines and a compact form factor. Available in colors like Midnight Black, Ocean
#  Blue, and Pearl White, it's a head-turner on the road.
#  ●Interior Design: Inside, the EcoSprint is a realm of comfort and luxury. It offers a
#  spacious cabin with seating for five, premium upholstery, and customizable ambient
#  lighting.
#  3. Performance Specifications
#  ●Engine and Motor: Powered by a high-efficiency electric motor, the EcoSprint delivers
#  200 horsepower and 300 Nm of torque, providing a smooth and responsive driving
#  experience.
#  ●Battery and Range: Equipped with a 50 kWh lithium-ion battery, it offers an impressive
#  range of up to 250 miles on a single charge. Charging is convenient with options for
#  home-charging setups and public charging stations.
#  ●Acceleration and Top Speed: The vehicle accelerates from 0 to 60 mph in just 7.3
#  seconds and has a top speed of 120 mph.
#  4. Technology and Features
#  ●Infotainment System: A state-of-the-art infotainment system with a 10-inch touchscreen,
#  voice control, and smartphone integration for both Android and iOS devices.
#  ●Driver Assistance Systems: Includes advanced features like adaptive cruise control,
#  lane-keeping assist, and automatic emergency braking.
#  ●Connectivity: Remote monitoring and control via a smartphone app, enabling
#  pre-conditioning of the vehicle, charging status checks, and more.
#  5. Safety and Security
#  ●Safety Features: High-rated safety features including multiple airbags, a reinforced
#  frame, and blind-spot monitoring.
#  ●Security Features: Advanced security systems with remote locking/unlocking, GPS
#  tracking, and intrusion alarms.
#********************************************************
#  > Reflection: {'is_done': False, 'feedback': 'The assistant has not provided a summary of the product specification with less than 100 words, focusing on performance specifications and safety features. The assistant has only copied the entire specification document without summarizing it.'}
#********************************************************
#  Correction: The EcoSprint is an electric vehicle with a high-efficiency motor, delivering 200 horsepower and 300 Nm of torque. It features advanced safety features, including multiple airbags and blind-spot monitoring, and has a range of up to 250 miles on a single charge.
#********************************************************
#  > Running step 059c031f-7c58-41d8-84da-c606f0e95ffc. Step input: None
#********************************************************
#  > Reflection: {'is_done': False, 'feedback': 'The assistant has provided a summary of the product specification with less than 100 words, focusing on performance specifications and safety features. However, the task is not fully completed as the final message is not an assistant message that confirms the task is done.'}
#  Correction: The EcoSprint is an electric vehicle with a high-efficiency motor, delivering 200 horsepower and 300 Nm of torque. It features advanced safety features, including multiple airbags and blind-spot monitoring, and has a range of up to 250 miles on a single charge. This information confirms the task is done.
#  > Running step a01dd4d0-1d5d-4341-9d3c-5d032291ae07. Step input: None
#  > Reflection: {'is_done': True, 'feedback': 'The assistant has provided a summary of the product specification with less than 100 words, focusing on performance specifications and safety features, and the final message is an assistant message that confirms the task is done.'}
#  
#   The EcoSprint is an electric vehicle with a high-efficiency motor, delivering 200 horsepower and 300 Nm of torque. It features advanced safety features, including multiple airbags and blind-spot monitoring, and has a range of up to 250 miles on a single charge. This information confirms the task is done.