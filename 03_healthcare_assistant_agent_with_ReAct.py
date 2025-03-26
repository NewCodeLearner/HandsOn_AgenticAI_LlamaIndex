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


# Set up wikipedia tools
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.tools.tool_spec.load_and_search import (
    LoadAndSearchToolSpec,
)
# Get list of wikipedia tools
wiki_spec = WikipediaToolSpec()
tool = wiki_spec.to_tool_list()[1]
wiki_tools = LoadAndSearchToolSpec.from_defaults(tool).to_tool_list()

# Review the list of tools
for tool in wiki_tools:
    print( "--------\n",tool.metadata.name, tool.metadata.description)

# Set up RAG for medications
medication_document =JSONReader().load_data(
    input_file='data/Medications and dosages.json'
)

splitter = SentenceSplitter(chunk_size=256)
medication_nodes = splitter.get_nodes_from_documents(medication_document)
medication_index = VectorStoreIndex(medication_nodes,show_progress=True)

# Create a query engine tool based on the CSV file
medication_query_engine = medication_index.as_query_engine()
medication_tool = QueryEngineTool.from_defaults(
    query_engine=medication_query_engine,
    description=(
        "Provides list of popular medications, diseases they are used to treat for and recommended dosages"
    )
    )

# Set up & execute the Healthcare Agent

context = """
You are an healthcare assistant who can answer questions regarding 
diseases, their symptoms and medications for them.

Use the wikipedia tools to answer questions about symptoms and possible 
diseases associated with those symptoms. 
Use the medication tool to find list of medications and dosages for a given
disease.

Use only the tools provided to answer questions and NOT your own memory.
"""

healthcare_tools =wiki_tools + [medication_tool]
healthcare_agent = ReActAgent.from_tools(
    tools = healthcare_tools,
    llm = Settings.llm,
    verbose = True,
    context = context
)

#response=healthcare_agent.chat("Which medication should i take for arthritis?")
#print("*******\n Response : ",response.response)

#********************************************
# ACTUAL RESPONSE FOR THE ABOVE QUERY
#********************************************
# > Running step 3f9d6181-fef3-41b1-a85c-28c3e47c2e4b. Step input: Which medication should i take for arthritis?
# Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
# Action: query_engine_tool
# Action Input: {'input': 'arthritis'}
# Observation: There is no information about arthritis in the given context.
# > Running step 40f599d9-6856-4259-8c53-92bf9e43d95c. Step input: None
# Thought: The tool did not have enough information about arthritis. I need to search for arthritis on Wikipedia to get more information about it.
# Action: search_data
# Action Input: {'query': 'arthritis', 'lang': 'en'}
# Observation: Content loaded! You can now search the information using read_search_data
# > Running step f5f73b11-6bf2-431c-9f0c-9def18452b69. Step input: None
# Thought: I have loaded the data about arthritis from Wikipedia. Now, I need to read the data to find out about the medications used to treat arthritis.
# Action: read_search_data
# Action Input: {'query': 'what medications are used to treat arthritis'}
# Observation: Medications used to treat arthritis include acetaminophen, non-steroidal anti-inflammatory drugs (NSAIDs) like ibuprofen, opioids, and intra-articular corticosteroid injections. For rheumatoid arthritis, treatments may also include disease-modifying antirheumatic drugs (DMARDs), such as csDMARDs, TNF biologics, and tsDMARDs, as well as corticosteroids and monoclonal antibodies given intravenously. Topical NSAIDs may also be used, which may have better safety profiles than oral NSAIDs.     
# > Running step 996f5a72-fe25-4624-811a-08bf4b43ea5b. Step input: None
# Thought: I have found the information about the medications used to treat arthritis. Now, I need to find out the recommended dosages for these medications.
# Action: query_engine_tool
# Action Input: {'input': 'dosage of acetaminophen, ibuprofen, and disease-modifying antirheumatic drugs for arthritis'}
# Observation: For Acetaminophen, the typical dosage is 500-1000 mg every 4-6 hours as needed. For Ibuprofen, the typical dosage is 200-400 mg every 4-6 hours as needed. However, there is no information provided about disease-modifying antirheumatic drugs or their dosage for arthritis.
# > Running step 20c1f901-a98f-4c9f-a4fa-b55f2986e11a. Step input: None
# Thought: I have found the information about the dosages of acetaminophen and ibuprofen for arthritis. However, I couldn't find the information about the disease-modifying antirheumatic drugs. I can still provide the user with the information I have.
# Answer: For arthritis, you can take acetaminophen (500-1000 mg every 4-6 hours as needed) or ibuprofen (200-400 mg every 4-6 hours as needed). However, please consult a 
# doctor for a personalized treatment plan, especially for disease-modifying antirheumatic drugs.
# *******
#  Response :  For arthritis, you can take acetaminophen (500-1000 mg every 4-6 hours as needed) or ibuprofen (200-400 mg every 4-6 hours as needed). However, please consult a doctor for a personalized treatment plan, especially for disease-modifying antirheumatic drugs.


response=healthcare_agent.chat("Which medication should i take for bacterial infection?")
print("*******\n Response : ",response.response)