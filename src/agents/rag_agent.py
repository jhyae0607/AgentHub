from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Literal
from typing import Annotated, List
from dataclasses import dataclass, field
from langchain_core.messages import AIMessage

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

from agents.utils import get_model_from_config, extract_message_content

# Initialization
num_of_docs_to_retrieve = 5
num_of_docs_to_use_for_generation = 5
max_iteration = 2
strip_thinking_tokens = True

# Prompts
grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

Your task is to determine if the document contains information that could help answer the user's question.

IMPORTANT RULES:
1. If the document contains ANY information related to the question's topic, grade it as 'yes'
2. Only grade as 'no' if the document is completely unrelated to the question
3. Be lenient in your grading - it's better to keep a potentially relevant document than to filter it out
4. You must respond with EXACTLY 'yes' or 'no' (lowercase)

Example:
Question: "What is machine learning?"
Document: "Machine learning is a subset of artificial intelligence..."
Grade: yes

Question: "What is machine learning?"
Document: "The weather forecast for tomorrow..."
Grade: no

Now grade the following document for relevance to the question."""

relevancy_checker_instructions = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


query_rewriter_instructions = """You are a question re-writer that converts an input question to a better version that is optimized \n 
for vectorstore retrieval. Do not change the original meaning of the question and only output the rewritten question."""

generate_instructions = """You are an AI assistant specializing in Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system. 
Your primary mission is to answer questions based on provided context or chat history.
Ensure your response is concise and directly addresses the question without any additional narration.

###

Your final answer should be written concisely (but include important numerical values, technical terms, jargon, and names), followed by the source of the information.

# Steps

1. Carefully read and understand the context provided.
2. Identify the key information related to the question within the context.
3. Formulate a concise answer based on the relevant information.
4. Ensure your final answer directly addresses the question.
5. List the source of the answer in bullet points, which must be a file name (with a page number) or URL from the context. Omit if the source cannot be found.

# Output Format:
[Your final answer here, with numerical values, technical terms, jargon, and names in their original language]

**Source**(REQUIRED)
- (Source of the answer, must be a file name(with a page number) or URL from the context. Omit if you can't find the source of the answer.)
- (list more if there are multiple sources)
- ...

###

Remember:
- It's crucial to base your answer solely on the **PROVIDED CONTEXT**. 
- DO NOT use any external knowledge or information not present in the given materials.
- If you can't find the source of the answer, you should answer that you don't know.
- Ensure to include the source metadataof the answer in the final answer.
###

# Here is the user's QUESTION that you should answer:
{question}

# Here is the CONTEXT that you should use to answer the question:
{context}

# Your final ANSWER to the user's QUESTION:
"""

@dataclass
class RetrievalState:
    """
    Data model for graph state

    Attributes:
        messages: user input message
        search_query: search query for vectorstore retrieval
        generation: LLM generated answer
        documents: document list
        iteration: current iteration
    """
    messages: Annotated[str, "User query"] = field(default=None)
    search_query: Annotated[str, "Search query"] = field(default=None)
    generation: Annotated[str, "LLM generated answer"] = field(default=None)
    documents: Annotated[List[str], "List of documents"] = field(default_factory=list)
    iteration: Annotated[int, "Current iteration"] = field(default=0)


# Add input and output state classes
@dataclass
class RetrievalStateInput:
    messages: str = field(default=None)

@dataclass
class RetrievalStateOutput:
    generation: str = field(default=None)
    messages: List[AIMessage] = field(default_factory=list)

# Retrieve Node
async def retrieve_documents(state:RetrievalState, config:RunnableConfig):
    question = extract_message_content(state.messages)

    state.search_query = question

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="joint_planning"  # The name you used when creating the vectorstore
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": num_of_docs_to_retrieve
        }
    )

    new_documents = retriever.invoke(state.search_query)
    
    status_message = f"🔍 Retrieving relevant documents...\n\nFound {len(new_documents)} documents in iteration {state.iteration + 1}"
    
    if state.iteration == 0:
        state.documents = new_documents
        state.iteration += 1
        return {
            "documents": state.documents,
            "iteration": state.iteration,
            "messages": [AIMessage(content=status_message)],
            "search_query": state.search_query
        }
    
    else:
        existing_doc_ids = {
            f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
            for doc in state.documents
        }
    
        unique_new_docs = [
            doc for doc in new_documents
            if f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}" not in existing_doc_ids
        ]
        
        status_message = f"🔍 Retrieving additional documents...\n\nFound {len(unique_new_docs)} new unique documents in iteration {state.iteration + 1}"
        
        state.documents.extend(unique_new_docs)
        state.iteration += 1
        return {
            "documents": state.documents,
            "iteration": state.iteration,
            "messages": [AIMessage(content=status_message)],
            "search_query": state.search_query
        }

# Grade Documents Node
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="The binary score of the document. YES if the document is relevant, NO if it is not."
    )


def grade_documents(state:RetrievalState, config:RunnableConfig):
    model = get_model_from_config(config)
    structured_llm_grader = model.with_structured_output(GradeDocuments)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grader_instructions),
            ("human", "Retrieved document: \n\n {document} \n\n User Question: {question}")
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    filtered_docs = []
    for doc in state.documents:
        score = retrieval_grader.invoke(
            {"question": state.search_query, "document": doc.page_content}
        )
        grade = score.binary_score.lower()
        if grade == "yes":
            filtered_docs.append(doc)

    status_message = f"📊 Grading document relevance...\n\nKept {len(filtered_docs)} out of {len(state.documents)} documents as relevant to the query"

    state.documents = filtered_docs
    
    return {
        "documents": state.documents,
        "messages": [AIMessage(content=status_message)],
        "search_query": state.search_query
    }


# Decide to Generate Node
def decide_to_generate(state:RetrievalState, config:RunnableConfig) -> Literal["generate", "transform_query"]:
    if len(state.documents) >= num_of_docs_to_use_for_generation or state.iteration >= max_iteration:
        return "generate"
    else:
        return "transform_query"


# Generate Node
def format_docs(docs):
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata["source"]}</source><page>{doc.metadata["page"]+1}</page></document>'
            for doc in docs
        ]
    )

def generate(state:RetrievalState, config:RunnableConfig):
    model = get_model_from_config(config)

    formatted_docs = format_docs(state.documents)

    prompt = ChatPromptTemplate.from_template(generate_instructions)
    rag_chain = prompt | model | StrOutputParser()

    state.generation = rag_chain.invoke({"context": formatted_docs, "question": state.search_query})

    return {
        "generation": state.generation,
        "messages": [AIMessage(content=state.generation)],  # Only return the final generation
        "search_query": state.search_query
    }


# Transform Query Node
def transform_query(state:RetrievalState, config:RunnableConfig):
    model = get_model_from_config(config)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", query_rewriter_instructions),
            ("human", "Here is the initial query: \n\n {question} \n Formulate an improved version of the query that is optimized for vector retrieval.")
        ]
    )
    query_rewriter = prompt | model | StrOutputParser()
    better_query = query_rewriter.invoke({"question": state.search_query})

    status_message = f"🔄 Transforming query for better retrieval...\n\nOriginal: {state.search_query}\n\nImproved: {better_query}"

    state.search_query = better_query
    return {
        "messages": [AIMessage(content=status_message)],
        "search_query": state.search_query
    }


# Cleanup Node
def cleanup_state(state:RetrievalState, config:RunnableConfig):
        # Reset all state variables
    state.messages = None
    state.search_query = None
    state.generation = None
    state.documents = []
    state.iteration = 0
    
    return {
        "messages": [],
        "search_query": None,
        "generation": None,
        "documents": [],
        "iteration": 0
    }

# Compile the graph
workflow = StateGraph(RetrievalState, input=RetrievalStateInput, output=RetrievalStateOutput)

workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)
workflow.add_node("cleanup", cleanup_state)

workflow.add_edge(START, "retrieve_documents")
workflow.add_edge("retrieve_documents", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents", 
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query"
    }
)

workflow.add_edge("transform_query", "retrieve_documents")
workflow.add_edge("generate", "cleanup")
workflow.add_edge("cleanup", END)

rag_agent = workflow.compile(
    checkpointer=MemorySaver(),
    store=InMemoryStore(),
)
