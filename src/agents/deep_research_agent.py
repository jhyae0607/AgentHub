import json
import operator
from datetime import datetime
from dataclasses import dataclass, field
from typing_extensions import Literal, Annotated

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

from agents.utils import (
    deduplicate_and_format_sources, 
    format_sources, 
    strip_thinking_tokens,
    get_model_from_config,
    extract_message_content
)
from agents.tools import tavily_search_func

# Initialization
max_web_search_loop = 2
tavily_search_max_results = 3

# State Classes
@dataclass(kw_only=True)
class SummaryState:
    messages: str = field(default=None)    
    search_query: str = field(default=None)
    web_research_results: Annotated[list, operator.add] = field(default_factory=list) 
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list) 
    research_loop_count: int = field(default=0)
    running_summary: str = field(default=None)

@dataclass(kw_only=True)
class SummaryStateInput:
    messages: str = field(default=None)

@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None) 

# Prompts
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

query_writer_instructions="""Your goal is to generate a targeted web search query.

<CONTEXT>
Current date: {current_date}
Please ensure your queries account for the most current information available as of this date.
</CONTEXT>

<TOPIC>
{search_query}
</TOPIC>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "machine learning transformer architecture explained",
    "rationale": "Understanding the fundamental structure of transformer models"
}}
</EXAMPLE>

Provide your response in JSON format:"""

summarizer_instructions="""
<GOAL>
Generate a high-quality summary of the provided context.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.                                                                                                                                                            
</REQUIREMENTS>

<FORMATTING>
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
</FORMATTING>

<Task>
Think carefully about the provided Context first. Then generate a summary of the context to address the User Input.
</Task>
"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {messages}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for web search.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}
</Task>

Provide your analysis in JSON format:"""

# Graph Nodes
async def generate_query(state: SummaryState, config: RunnableConfig):
    model = get_model_from_config(config)
    
    messages = extract_message_content(state.messages)
    
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        search_query=messages
    )
    
    result = model.invoke(
        [SystemMessage(content=formatted_prompt),
        HumanMessage(content="Generate a query for web search:")]
    )
    
    content = result.content

    try:
        query = json.loads(content)
        search_query = query['query']
        rationale = query.get('rationale', '')
    except (json.JSONDecodeError, KeyError):
        content = strip_thinking_tokens(content)
        search_query = content
        rationale = "Generated query based on user input"
        
    status_message = f"🔍 Generating search query...\n\nQuery: {search_query}\n\nReason: {rationale}"
        
    return {
        "search_query": search_query,
        "messages": [AIMessage(content=status_message)]
    }


async def web_research(state: SummaryState, config: RunnableConfig):
    search_results = tavily_search_func(state.search_query, fetch_full_page=True, max_results=tavily_search_max_results)
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=True)

    status_message = f"🌐 Searching the web for: {state.search_query}\n\nFound {len(search_results)} relevant sources."
    
    return {
        "sources_gathered": [format_sources(search_results)], 
        "research_loop_count": state.research_loop_count + 1, 
        "web_research_results": [search_str],
        "messages": [AIMessage(content=status_message)]
    }

async def summarize_sources(state: SummaryState, config: RunnableConfig):
    model = get_model_from_config(config)
    
    existing_summary = state.running_summary

    most_recent_web_research = state.web_research_results[-1]

    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {most_recent_web_research} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {state.messages} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_web_research} \n <Context>"
            f"Create a Summary using the Context on this topic: \n <User Input> \n {state.messages} \n <User Input>\n\n"
        )

    result = model.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result.content
    running_summary = strip_thinking_tokens(running_summary) # for deepseek models

    status_message = f"📝 {'Updating' if existing_summary else 'Creating'} summary...\n\n{running_summary}"

    return {
        "running_summary": running_summary,
        "messages": [AIMessage(content=status_message)]
    }

async def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    model = get_model_from_config(config)
    
    result = model.invoke(
        [SystemMessage(content=reflection_instructions.format(messages=state.messages)),
        HumanMessage(content=f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n And now identify a knowledge gap and generate a follow-up web search query:")]
    )
    
    try:
        reflection_content = json.loads(result.content)
        query = reflection_content.get('follow_up_query')
        knowledge_gap = reflection_content.get('knowledge_gap', '')
        if not query:
            query = f"Tell me more about {state.search_query}"
            knowledge_gap = "General information about the topic"
    except (json.JSONDecodeError, KeyError, AttributeError):
        query = f"Tell me more about {state.search_query}"
        knowledge_gap = "General information about the topic"

    status_message = f"🤔 Analyzing knowledge gaps...\n\nIdentified Gap: {knowledge_gap}\n\nNext Query: {query}"

    return {
        "search_query": query,
        "messages": [AIMessage(content=status_message)]
    }

async def finalize_summary(state: SummaryState):
    current_sources = state.sources_gathered[-1] if state.sources_gathered else []
    
    seen_sources = set()
    unique_sources = []
    
    for line in current_sources.split('\n'):
        if line.strip() and line not in seen_sources:
            seen_sources.add(line)
            unique_sources.append(line)
    
    all_sources = "\n".join(unique_sources)
    final_summary = f"{state.running_summary}\n\n### Sources:\n{all_sources}"
    
    status_message = f"📚 Finalizing search report...\n\n{final_summary}"
    
    return {
        "running_summary": final_summary,
        "messages": [AIMessage(content=status_message)],
        "sources_gathered": []
    }

def cleanup_state(state: SummaryState, config: RunnableConfig):
    state.messages = None
    state.search_query = None
    state.web_research_results = []
    state.sources_gathered = []
    state.research_loop_count = 0
    state.running_summary = None

    return {
        "messages": [AIMessage(content="Research completed. State has been reset.")],
        "search_query": None,
        "web_research_results": [],
        "sources_gathered": [],
        "research_loop_count": 0,
        "running_summary": None
    }

async def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "reflect_on_summary"]:
    if state.research_loop_count >= max_web_search_loop:
        return "finalize_summary"
    else:
        return "reflect_on_summary"

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)
builder.add_node("cleanup", cleanup_state)

builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_conditional_edges(
    "summarize_sources", 
    route_research,
    {
        "finalize_summary": "finalize_summary",
        "reflect_on_summary": "reflect_on_summary"
    }
)
builder.add_edge("reflect_on_summary", "generate_query")
builder.add_edge("finalize_summary", "cleanup")
builder.add_edge("cleanup", END)

deep_research_agent = builder.compile(
    checkpointer=MemorySaver(),
    store=InMemoryStore(),
)