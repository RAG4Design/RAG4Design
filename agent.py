import os
import config
from pattool import PatentSearchTool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from rag import design_assistant
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
import base64
import requests
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from typing import Literal

os.environ["OPENAI_API_KEY"] = config.openai_api_key

tools = [PatentSearchTool()]


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=config.openai_model_name, temperature=config.temperature)

llm_query = llm.bind_tools(tools)


def query_patent(state: State):
    print("state: ", state)
    return {"messages": [llm_query.invoke(state["messages"])]}


graph_builder.add_node("query_patent", query_patent)


def rag_generate(state: State):
    return {"messages": [design_assistant.invoke({"content": config.HUMAN_PROMPT})]}


graph_builder.add_node("rag_generate", rag_generate)


def generate_design_draft(state: State):
    return {
        "messages": [
            AIMessage(content=
                base64.b64encode(
                    requests.get(
                        DallEAPIWrapper().run(
                            llm.invoke(
                                {
                                    "content": f"{config.IMAGE_GEN_PROMPT} {state['messages'][-1].content}"
                                }
                            )
                        )
                    ).content
                )
            )
        ]
    }


query_tool_node = ToolNode(tools=tools)
graph_builder.add_node("query_tool", query_tool_node)


def route_query_tool(
    state: State,
) -> Literal["query_tool", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if (
        "finish_reason" in ai_message.response_metadata
        and ai_message.response_metadata["finish_reason"] == "stop"
    ):
        return "rag_generate"
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "query_tool"
    return "rag_generate"


graph_builder.add_conditional_edges(
    "query_patent",
    route_query_tool,
)


graph_builder.add_edge(START, "query_patent")

graph_builder.add_edge("query_tool", "rag_generate")

graph = graph_builder.compile()


graph.invoke(
    {
        "messages": [
            SystemMessage(content=config.SYSTEM_PROMPT),
            HumanMessage(content=config.HUMAN_PROMPT),
        ]
    }
)
