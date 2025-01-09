from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import os
import config
from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
import torch

persist_dir = config.persist_dir


def get_vectorstore(persist_dir, embedding_function, force_recreate=False):
    if not os.path.exists(persist_dir) or force_recreate:
        print("creating vector store ...")
        vectorstore = Chroma(
            embedding_function=embedding_function, persist_directory=persist_dir
        )
    else:
        print(f"loading vector store from {persist_dir} ...")
        vectorstore = Chroma(
            persist_directory=persist_dir, embedding_function=embedding_function
        )

    return vectorstore


def device_select():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


device = device_select()

print(f"You are using your: {device}\nCUDA available: {torch.cuda.is_available()}")


model_name = config.embedding_model_name
model_kwargs = {"device": device}

embedding_function = SentenceTransformerEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

vectorstore = get_vectorstore(persist_dir, embedding_function, force_recreate=False)

retriever = vectorstore.as_retriever(search_kwargs={"k": config.retriever_k})

design_assistant = None
if config.openai_assistant_id is None:
    design_assistant = OpenAIAssistantRunnable.create_assistant(
        name="design assistant",
        instructions=config.ASSISTANT_INSTRUCTIONS,
        tools=[{"type": "file_search"}],
        model=config.openai_model_name,
    )
else:
    design_assistant = OpenAIAssistantRunnable(assistant_id=config.openai_assistant_id)
