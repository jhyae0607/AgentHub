from dataclasses import dataclass

from langgraph.pregel import Pregel

from agents.chatbot import chatbot
from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.deep_research_agent import deep_research_agent
from agents.rag_agent import rag_agent
from agents.mcp_agent import mcp_agent
from schema import AgentInfo

DEFAULT_AGENT = "mcp-agent"


@dataclass
class Agent:
    description: str
    graph: Pregel


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot.", graph=chatbot),
    "deep-research-assistant": Agent(
        description="A deep research assistant with iterative web search.", graph=deep_research_agent
    ),
    "rag-agent": Agent(
        description="A RAG agent with access to information in a database.", graph=rag_agent
    ),
    "langgraph-supervisor-agent": Agent(
        description="A langgraph supervisor agent", graph=langgraph_supervisor_agent
    ),
    "mcp-agent": Agent(
        description="A MCP agent with access to different tools and resources.", graph=mcp_agent
    ),
}


def get_agent(agent_id: str) -> Pregel:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
