"#Handson Agentic AI using LlamaIndex"


```markdown
# Hands-On Agentic AI with LlamaIndex

This repository provides a series of Python scripts that explore the capabilities of [LlamaIndex](https://github.com/jerryjliu/llama_index) in building agentic AI applications. Each script demonstrates a unique use case, offering a hands-on approach to understanding and implementing agent-based systems using LlamaIndex.

## Prerequisites

Before running the scripts, ensure you have the following installed:

- Python 3.x
- Required Python packages (see [Installation](#installation))
- Make sure to create Groq - GROQ API KEY and add it to .env file, as the LLM model is used by calling Groq API. Groq link - https://console.groq.com/docs/overview

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NewCodeLearner/HandsOn_AgenticAI_LlamaIndex.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd HandsOn_AgenticAI_LlamaIndex
   ```
3. **Install the required packages:**
   The dependencies are listed in the `requirements.txt` file. Install them using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Scripts Overview

The repository contains the following Python scripts, each prefixed with a number indicating the suggested order of execution:

1. **`01_simple_rag.py`**: Introduces a basic Retrieval-Augmented Generation (RAG) setup using LlamaIndex.
2. **`02_basic_llamaindex_agent.py`**: Demonstrates the creation of a simple agent leveraging LlamaIndex for information retrieval and response generation.
3. **`03_healthcare_assistant_agent_with_ReAct.py`**: Builds a healthcare assistant agent utilizing the ReAct framework in conjunction with LlamaIndex for dynamic response generation.
4. **`04_summarization_with_reflection_agent.py`**: Explores document summarization techniques using a reflection-based agent powered by LlamaIndex.
5. **`05_simple_workflow.py`**: Illustrates the implementation of a straightforward workflow integrating LlamaIndex for task automation.
6. **`06_ReAct_agent_with_workflows_doctor_scheduling.py`**: Combines the ReAct framework with workflow automation to develop a doctor scheduling assistant agent.

## Usage

To run a specific script:

```bash
python <script_name>.py
```

For example, to execute the basic RAG setup:

```bash
python 01_simple_rag.py
```


Each script is designed to be self-contained, providing insights into different aspects of agentic AI using LlamaIndex. It's recommended to run the scripts in the order provided to build a comprehensive understanding.

## Data and Libraries

- **Data**: The `data` directory contains sample datasets used by the scripts. Ensure this directory is present and populated as required by each script.
- **Libraries**: Custom utility functions and modules are located in the `lib` directory, supporting the main scripts.

```
