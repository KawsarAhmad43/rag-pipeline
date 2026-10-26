from dotenv import load_dotenv
import os
from pathlib import Path
import requests
import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_tavily import TavilySearch
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from operator import itemgetter
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
import operator
import functools
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Ensure config.py exists and contains necessary variables
try:
    from config import (
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBEDDING_MODEL,
        COLLECTION_NAME,
        get_current_model,
        get_current_provider,
        set_current_model,
        get_available_models,
    )
except ImportError as e:
    print(f"Error importing from config: {e}")
    exit(1)

# Ensure tools.py exists and contains necessary functions
try:
    from tools import (
        create_outline,
        read_document,
        write_document,
        edit_document,
    )
except ImportError:
    print(
        "Error: tools.py not found or missing functions. "
        "Please ensure create_outline, read_document, write_document, "
        "edit_document are defined in tools.py"
    )
    exit(1)

from langchain_core.tools import tool
import tempfile

PROJECT_DIR = Path.cwd() / "llama_context_project"
DATA_DIR = PROJECT_DIR / "data"

# Global variable to store the current document path
current_document_path = None

def create_llm():
    """Create LLM based on current model selection"""
    current_model = get_current_model()
    current_provider = get_current_provider()

    if current_provider == "openai":
        return ChatOpenAI(model=current_model)
    elif current_provider == "gemini":
        return ChatGoogleGenerativeAI(model=current_model)
    else:
        # Fallback to OpenAI
        return ChatOpenAI(model="gpt-4o-mini")

def set_current_document(document_path: str):
    """Set the current document path for processing"""
    global current_document_path
    current_document_path = document_path

def get_current_document():
    """Get the current document path for processing"""
    global current_document_path
    if current_document_path and os.path.exists(current_document_path):
        return current_document_path
    return None

def _token_len(text: str) -> int:
    """Safe token length function that handles unknown model names"""
    try:
        current_model = get_current_model()
        enc = tiktoken.encoding_for_model(current_model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def load_and_split_documents(document_path: str):
    """Load and split documents from the provided path. Document path is required."""
    if not document_path:
        raise ValueError("Document path is required. Please upload a document first.")

    if not os.path.exists(document_path):
        raise ValueError(f"Document file not found: {document_path}")

    loader = PyMuPDFLoader(document_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_token_len,
    )
    return text_splitter.split_documents(docs)


def create_vectorstore(document_path: str):
    """Create vector store from the provided document path. Document path is required."""
    if not document_path:
        raise ValueError("Document path is required. Please upload a document first.")

    chunks = load_and_split_documents(document_path)
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Qdrant.from_documents(
        chunks,
        embedding_model,
        location=":memory:",
        collection_name=COLLECTION_NAME,
    )


def create_rag_chain(document_path: str):
    """Create RAG chain from the provided document path. Document path is required."""
    if not document_path:
        raise ValueError("Document path is required. Please upload a document first.")

    vectorstore = create_vectorstore(document_path)
    retriever = vectorstore.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(
        """
        CONTEXT:
        {context}

        QUERY:
        {question}

        You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
        """
    )
    llm = create_llm()
    return (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | rag_prompt
        | llm
        | StrOutputParser()
    )


@tool
def retrieve_information(query: str):
    """Use Retrieval Augmented Generation to retrieve information about the uploaded document."""
    global current_document_path
    if not current_document_path:
        return "No document has been uploaded yet. Please upload a PDF document first."

    if not os.path.exists(current_document_path):
        return "The uploaded document file is no longer available. Please upload the document again."

    try:
        # Validate the document path is still accessible
        if not os.path.isfile(current_document_path):
            return "The uploaded document is not a valid file. Please upload the document again."

        # Check file size to ensure it's not empty
        if os.path.getsize(current_document_path) == 0:
            return "The uploaded document appears to be empty. Please upload a valid document."

        return create_rag_chain(current_document_path).invoke({"question": query})
    except Exception as e:
        return f"Error processing document: {str(e)}. Please try uploading the document again."


def create_simple_rag_chain():
    """Create a simple RAG chain for basic queries without multi-agent complexity"""
    global current_document_path
    if not current_document_path:
        return None

    try:
        vectorstore = create_vectorstore(current_document_path)
        retriever = vectorstore.as_retriever()
        rag_prompt = ChatPromptTemplate.from_template(
            """
            CONTEXT:
            {context}

            QUERY:
            {question}

            You are a helpful assistant. Use the available context to answer the question.
            Keep your response concise and focused. If you can't answer the question, say you don't know.
            """
        )
        llm = create_llm()
        return (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
    except Exception as e:
        return None


def is_simple_query(message: str) -> bool:
    """Determine if a query is simple enough to use the simple RAG chain"""
    simple_keywords = [
        "summarize",
        "summary",
        "what is",
        "explain",
        "describe",
        "brief",
        "overview",
        "main points",
        "key points",
        "about",
        "content",
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in simple_keywords)


def create_agent(llm, tools, system_prompt):
    """Create an agent with the given LLM, tools, and system prompt"""
    system_prompt += (
        "\nWork autonomously according to your specialty, using the tools available to you."
        " Do not ask for clarification."
        " Your other team members (and other teams) will collaborate with you with their own specialties."
        " You are chosen for a reason! You are one of the following team members: {team_members}."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


def create_team_supervisor(llm, system_prompt, members):
    options = ["FINISH"] + members

    # Create a tool for routing
    @tool
    def route(next: str) -> str:
        """Select the next role to act."""
        if next not in options:
            return f"Invalid option. Must be one of: {options}"
        return f"Selected: {next}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),  # System prompt now includes dynamic date/time/rules
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}"
                "\n\nRemember: If the task appears complete or if you've made sufficient progress, respond with FINISH."
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))

    # Custom parser to extract the "next" value from tool calls
    def extract_next_value(tool_calls):
        if tool_calls and len(tool_calls) > 0:
            # Get the first tool call
            tool_call = tool_calls[0]
            # Extract the "next" argument
            if "args" in tool_call and "next" in tool_call["args"]:
                return {"next": tool_call["args"]["next"]}
        # Fallback to FINISH if no valid tool call
        return {"next": "FINISH"}

    return (
        prompt
        | llm.bind_tools([route])
        | JsonOutputToolsParser()
        | extract_next_value
    )


def agent_node(state, agent, name):
    try:
        result = agent.invoke(state)
        return {"messages": [HumanMessage(content=result["output"], name=name)]}
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error in {name}: {str(e)}"
        return {"messages": [HumanMessage(content=error_message, name=name)]}


class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str


class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str


def prelude(state):
    written_files = []
    if not PROJECT_DIR.exists():
        PROJECT_DIR.mkdir()
    try:
        written_files = [f.relative_to(PROJECT_DIR) for f in PROJECT_DIR.rglob("*")]
    except:
        pass
    if not written_files:
        return {**state, "current_files": "No files written."}
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files]),
    }


def create_research_graph():
    llm = create_llm()
    tavily_tool = TavilySearch(max_results=3, handle_tool_error=True)  # Reduced max_results
    search_agent = create_agent(
        llm,
        [tavily_tool],
        "You are a research assistant who can search for up-to-date info using the tavily search engine. Keep your responses concise and focused.",
    )
    research_agent = create_agent(
        llm,
        [retrieve_information],
        "You are a research assistant who can provide specific information on the uploaded document. You must only respond with information about the uploaded document related to the request. If no document is uploaded, inform the user to upload a document first. Keep your responses concise and focused.",
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    research_node = functools.partial(
        agent_node, agent=research_agent, name="PaperInformationRetriever"
    )
    supervisor_agent = create_team_supervisor(
        llm,
        (
            "You are a supervisor tasked with managing a conversation between the"
            " following workers: Search, PaperInformationRetriever. Given the following user request,"
            " determine the subject to be researched and respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. "
            " You should never ask your team to do anything beyond research. They are not required to write content or posts."
            " You should only pass tasks to workers that are specifically research focused."
            " IMPORTANT: After 2-3 rounds of research, you MUST respond with FINISH to avoid infinite loops."
        ),
        ["Search", "PaperInformationRetriever"],
    )
    research_graph = StateGraph(ResearchTeamState)
    research_graph.add_node("Search", search_node)
    research_graph.add_node("PaperInformationRetriever", research_node)
    research_graph.add_node("supervisor", supervisor_agent)
    research_graph.add_edge("Search", "supervisor")
    research_graph.add_edge("PaperInformationRetriever", "supervisor")
    research_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "Search": "Search",
            "PaperInformationRetriever": "PaperInformationRetriever",
            "FINISH": END,
        },
    )
    research_graph.set_entry_point("supervisor")
    return research_graph.compile()


def create_authoring_graph():
    llm = create_llm()
    doc_writer_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        "You are an expert writing technical LinkedIn posts. Keep your responses concise and focused.\nBelow are files currently in your directory:\n{current_files}",
    )
    note_taking_agent = create_agent(
        llm,
        [create_outline, read_document],
        "You are an expert senior researcher tasked with writing a LinkedIn post outline and taking notes to craft a LinkedIn post. Keep your responses concise and focused.\n{current_files}",
    )
    copy_editor_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        "You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues. Keep your responses concise and focused.\nBelow are files currently in your directory:\n{current_files}",
    )
    dopeness_editor_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        "You are an expert in dopeness, litness, coolness, etc - you edit the document to make sure it's dope. Make sure to use a number of emojis. Keep your responses concise and focused.\nBelow are files currently in your directory:\n{current_files}",
    )
    context_aware_doc_writer_agent = prelude | doc_writer_agent
    doc_writing_node = functools.partial(
        agent_node, agent=context_aware_doc_writer_agent, name="DocWriter"
    )
    context_aware_note_taking_agent = prelude | note_taking_agent
    note_taking_node = functools.partial(
        agent_node, agent=context_aware_note_taking_agent, name="NoteTaker"
    )
    context_aware_copy_editor_agent = prelude | copy_editor_agent
    copy_editing_node = functools.partial(
        agent_node, agent=context_aware_copy_editor_agent, name="CopyEditor"
    )
    context_aware_dopeness_editor_agent = prelude | dopeness_editor_agent
    dopeness_node = functools.partial(
        agent_node, agent=context_aware_dopeness_editor_agent, name="DopenessEditor"
    )
    doc_writing_supervisor = create_team_supervisor(
        llm,
        (
            "You are a supervisor tasked with managing a conversation between the"
            " following workers: {team_members}. You should always verify the technical"
            " contents after any edits are made. "
            "Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. "
            " IMPORTANT: After 3-4 rounds of work, you MUST respond with FINISH to avoid infinite loops."
        ),
        ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"],
    )
    authoring_graph = StateGraph(DocWritingState)
    authoring_graph.add_node("DocWriter", doc_writing_node)
    authoring_graph.add_node("NoteTaker", note_taking_node)
    authoring_graph.add_node("CopyEditor", copy_editing_node)
    authoring_graph.add_node("DopenessEditor", dopeness_node)
    authoring_graph.add_node("supervisor", doc_writing_supervisor)
    authoring_graph.add_edge("DocWriter", "supervisor")
    authoring_graph.add_edge("NoteTaker", "supervisor")
    authoring_graph.add_edge("CopyEditor", "supervisor")
    authoring_graph.add_edge("DopenessEditor", "supervisor")
    authoring_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "DocWriter": "DocWriter",
            "NoteTaker": "NoteTaker",
            "CopyEditor": "CopyEditor",
            "DopenessEditor": "DopenessEditor",
            "FINISH": END,
        },
    )
    authoring_graph.set_entry_point("supervisor")
    return authoring_graph.compile()


def create_super_graph():
    llm = create_llm()

    # Get current date and time
    current_datetime = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S %Z%z"
    )  # e.g., 2025-08-13 18:00:00 GMT+0600

    supervisor_node = create_team_supervisor(
        llm,
        (
            f"Current Date and Time: {current_datetime}\n\n"
            "You are a sophisticated AI supervisor tasked with managing a conversation between the"
            " following expert teams: {team_members}. Your primary goal is to provide the most accurate, relevant, and helpful response to the user's request."
            "\n\nThink step-by-step:"
            " 1. Understand the user's query and its intent."
            " 2. Assess if the query is a simple greeting (e.g., 'hello', 'hi', 'how are you?', 'good morning') or a very general, trivial question that does not require specialized team tools or specific information. If so, select FINISH, as no specialized action is needed."
            " 3. Check if a document has been uploaded (using the 'get_current_document()' function internally for awareness, but don't explicitly call it in the prompt). If a document is contextually present and the query appears to be related to its content (e.g., 'summarize the paper', 'what does the document say about X', 'explain Y from the PDF', 'tell me about the main findings in this document'), always prioritize routing to the 'Research team' (specifically the 'PaperInformationRetriever' agent within it) to retrieve information *specifically from the document*."
            " 4. If no document is uploaded, or if the query is a general knowledge question (e.g., 'tell me about America', 'what is Y', 'explain Z', 'what is the capital of France') that doesn't directly relate to an uploaded document, route to the 'Research team' (specifically the 'Search' agent within it) to gather information from the web."
            " 5. If the query clearly indicates a need for content creation, outlining, or editing (e.g., 'write a LinkedIn post about...', 'draft an outline for...', 'edit this text'), route to the 'LinkedIn team'."
            " 6. If the query is ambiguous, unclear, or you genuinely believe more information is needed from the user to proceed effectively with any specialized task, select FINISH. This signals that you need more input from the user."
            " 7. When all necessary information has been gathered and tasks completed, you must respond with FINISH."
        ),
        ["Research team", "LinkedIn team"],
    )

    class State(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str

    def get_last_message(state: State) -> str:
        return state["messages"][-1].content

    def join_graph(response: dict):
        return {"messages": [response["messages"][-1]]}

    research_chain = create_research_graph()
    authoring_chain = create_authoring_graph()

    def enter_research_chain(message: str):
        return {
            "messages": [HumanMessage(content=message)],
            "team_members": ["Search", "PaperInformationRetriever"],
        }

    def enter_authoring_chain(message: str):
        return {
            "messages": [HumanMessage(content=message)],
            "team_members": ", ".join(
                ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"]
            ),
        }

    research_graph_with_entry = enter_research_chain | research_chain
    authoring_graph_with_entry = enter_authoring_chain | authoring_chain

    super_graph = StateGraph(State)
    super_graph.add_node(
        "Research team", get_last_message | research_graph_with_entry | join_graph
    )
    super_graph.add_node(
        "LinkedIn team", get_last_message | authoring_graph_with_entry | join_graph
    )
    super_graph.add_node("supervisor", supervisor_node)
    super_graph.add_edge("Research team", "supervisor")
    super_graph.add_edge("LinkedIn team", "supervisor")
    super_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "LinkedIn team": "LinkedIn team",
            "Research team": "Research team",
            "FINISH": END,
        },
    )
    super_graph.set_entry_point("supervisor")
    return super_graph.compile()
