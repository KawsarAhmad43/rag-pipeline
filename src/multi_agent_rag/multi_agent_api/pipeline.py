from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import requests
import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_tavily import TavilySearch  # Updated import
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from operator import itemgetter
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
import operator
import functools
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, COLLECTION_NAME, LLM_MODEL
from tools import retrieve_information, create_outline, read_document, write_document, edit_document

PROJECT_DIR = Path.cwd() / "llama_context_project"
DATA_DIR = PROJECT_DIR / "data"

def download_pdf():
    pdf_url = "https://arxiv.org/pdf/2404.19553"
    pdf_path = DATA_DIR / "2404.19553.pdf"
    if not pdf_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        response = requests.get(pdf_url)
        with open(pdf_path, "wb") as f:
            f.write(response.content)
    return str(pdf_path)

def load_and_split_documents():
    pdf_path = download_pdf()
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda text: len(tiktoken.encoding_for_model(LLM_MODEL).encode(text)),
    )
    return text_splitter.split_documents(docs)

def create_vectorstore():
    chunks = load_and_split_documents()
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Qdrant.from_documents(
        chunks,
        embedding_model,
        location=":memory:",
        collection_name=COLLECTION_NAME,
    )

def create_rag_chain():
    vectorstore = create_vectorstore()
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
    llm = ChatOpenAI(model=LLM_MODEL)
    return (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

def create_agent(llm, tools, system_prompt):
    system_prompt += (
        "\nWork autonomously according to your specialty, using the tools available to you."
        " Do not ask for clarification."
        " Your other team members (and other teams) will collaborate with you with their own specialties."
        " You are chosen for a reason! You are one of the following team members: {team_members}.")
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
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                f"Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_tools(tools=[function_def], tool_choice={"type": "function", "function": {"name": "route"}})  # Updated tool_choice
        | JsonOutputFunctionsParser()
    )

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

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
    llm = ChatOpenAI(model=LLM_MODEL)
    tavily_tool = TavilySearch(max_results=5)  # Updated to use TavilySearch
    search_agent = create_agent(
        llm,
        [tavily_tool],
        "You are a research assistant who can search for up-to-date info using the tavily search engine.",
    )
    research_agent = create_agent(
        llm,
        [retrieve_information],
        "You are a research assistant who can provide specific information on the provided paper: 'Extending Llama-3’s Context Ten-Fold Overnight'. You must only respond with information about the paper related to the request.",
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    research_node = functools.partial(agent_node, agent=research_agent, name= "PaperInformationRetriever")
    supervisor_agent = create_team_supervisor(
        llm,
        (
            "You are a supervisor tasked with managing a conversation between the"
            " following workers: Search, PaperInformationRetriever. Given the following user request,"
            " determine the subject to be researched and respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. "
            " You should never ask your team to do anything beyond research. They are not required to write content or posts."
            " You should only pass tasks to workers that are specifically research focused."
            " When finished, respond with FINISH."
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
        {"Search": "Search", "PaperInformationRetriever": "PaperInformationRetriever", "FINISH": END},
    )
    research_graph.set_entry_point("supervisor")
    return research_graph.compile()

def create_authoring_graph():
    llm = ChatOpenAI(model=LLM_MODEL)
    doc_writer_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        "You are an expert writing technical LinkedIn posts.\nBelow are files currently in your directory:\n{current_files}",
    )
    note_taking_agent = create_agent(
        llm,
        [create_outline, read_document],
        "You are an expert senior researcher tasked with writing a LinkedIn post outline and taking notes to craft a LinkedIn post.\n{current_files}",
    )
    copy_editor_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        "You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues\nBelow are files currently in your directory:\n{current_files}",
    )
    dopeness_editor_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        "You are an expert in dopeness, litness, coolness, etc - you edit the document to make sure it's dope. Make sure to use a number of emojis.\nBelow are files currently in your directory:\n{current_files}",
    )
    context_aware_doc_writer_agent = prelude | doc_writer_agent
    doc_writing_node = functools.partial(agent_node, agent=context_aware_doc_writer_agent, name="DocWriter")
    context_aware_note_taking_agent = prelude | note_taking_agent
    note_taking_node = functools.partial(agent_node, agent=context_aware_note_taking_agent, name="NoteTaker")
    context_aware_copy_editor_agent = prelude | copy_editor_agent
    copy_editing_node = functools.partial(agent_node, agent=context_aware_copy_editor_agent, name="CopyEditor")
    context_aware_dopeness_editor_agent = prelude | dopeness_editor_agent
    dopeness_node = functools.partial(agent_node, agent=context_aware_dopeness_editor_agent, name="DopenessEditor")
    doc_writing_supervisor = create_team_supervisor(
        llm,
        (
            "You are a supervisor tasked with managing a conversation between the"
            " following workers: {team_members}. You should always verify the technical"
            " contents after any edits are made. "
            "Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When each team is finished,"
            " you must respond with FINISH."
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
    llm = ChatOpenAI(model=LLM_MODEL)
    supervisor_node = create_team_supervisor(
        llm,
        (
            "You are a supervisor tasked with managing a conversation between the"
            " following teams: {team_members}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When all workers are finished,"
            " you must respond with FINISH."
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

    super_graph = StateGraph(State)
    super_graph.add_node("Research team", get_last_message | create_research_graph() | join_graph)
    super_graph.add_node("LinkedIn team", get_last_message | create_authoring_graph() | join_graph)
    super_graph.add_node("supervisor", supervisor_node)
    super_graph.add_edge("Research team", "supervisor")
    super_graph.add_edge("LinkedIn team", "supervisor")
    super_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"LinkedIn team": "LinkedIn team", "Research team": "Research team", "FINISH": END},
    )
    super_graph.set_entry_point("supervisor")
    return super_graph.compile()
