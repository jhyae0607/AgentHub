from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.utils import get_model_from_config, load_mcp_config_json
from agents.tools import TOOLS


#Initialization
PATH_TO_MCP_CONFIG = "../mcp/mcp_config.json"
recursion_limit = 30

# State Classes
@dataclass
class InputState:
    messages: Sequence[AnyMessage] = field(default_factory=list)

@dataclass
class State(InputState):
    is_last_step: bool = field(default=False)

# Default Prompts
SYSTEM_PROMPT = """You are a helpful AI assistant. Use your tools to help the user with their tasks.

System time: {system_time}"""

# Nodes
async def make_graph(mcp_tools: Dict[str, Dict[str, str]], config: RunnableConfig):
    client = MultiServerMCPClient(mcp_tools)
    mcp_server_tools = await client.get_tools()
    model = get_model_from_config(config)
    # Combine MCP server tools with additional tools from tools.py
    all_tools = mcp_server_tools + TOOLS
    agent = create_react_agent(model, all_tools, checkpointer=MemorySaver())
    return agent

async def call_model(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our agent."""

    system_message = SYSTEM_PROMPT.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    mcp_json_path = PATH_TO_MCP_CONFIG
    mcp_tools_config = await load_mcp_config_json(mcp_json_path)
    mcp_tools = mcp_tools_config.get("mcpServers", {})
    # print(mcp_tools)

    response = None

    my_agent = await make_graph(mcp_tools, config)
    messages = [SystemMessage(content=system_message), *state.messages]
    response = cast(
        AIMessage,
        await my_agent.ainvoke({"messages": messages}, config),
    )

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    return {"messages": [response["messages"][-1]]}


def route_model_output(state: State) -> Literal["__end__", "tools"]:

    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    if not last_message.tool_calls:
        return "__end__"
    return "tools"

# Define a new graph
builder = StateGraph(State, input=InputState)
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge("__start__", "call_model")

builder.add_conditional_edges(
    "call_model",
    route_model_output,
)

builder.add_edge("tools", "call_model")

mcp_agent = builder.compile(
    checkpointer=MemorySaver(),
    store=InMemoryStore(),
)