# RAG4Design

## Get Started
### Configure Tokens, Keys and Paths
Specify API tokens, keys, and paths in [config.py](./config.py), and customize parameters including prompts, date range and temperature if needed

### Install Dependencies
Create virtual environment (optional)
```
python3 -m venv env
```

Install dependencies
```
pip3 install -r requirements.txt
```

### Execute
```
python3 agent.py
```

## Customize Workflow
The agent is defined in [agent.py](./agent.py) using LangGraph, and can be customized through adding nodes or tools in the definition.

The patent search is conducted with [PatSnap](https://www.patsnap.com/), and can be altered with the PatentSearchTool class in [pattool.py](./pattool.py).

RAG is currently with OpenAI's Assistant API and can be configured on OpenAI's web page or through OpenAI vector store API. The OpenAI vector store as well as a local vector store are defined in [rag.py](./rag.py).

Design draft generation utilizes OpenAI's Dall-E model through LangChain's wrapper defined as node in [agent.py](./agent.py).
