from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import List, TypedDict, Annotated
import functools
import operator
import re
from document_tools import DocumentTools

class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str

class AgentSystem:
    """Manages research and document writing teams."""

    def __init__(self, llm, rag_pipeline, document_tools):
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.document_tools = document_tools
        self.research_graph = self._create_research_graph()
        self.doc_writing_graph = self._create_doc_writing_graph()
        self.supervisor_graph = self._create_supervisor_graph()

    def create_agent(self, tools: list, system_prompt: str) -> Runnable:
        tool_names = ", ".join([tool.name for tool in tools])
        system_prompt += (
            f"\nYou have access to the following tools: {tool_names}. "
            "To use a tool, include its name in square brackets followed by the input, e.g., [tool_name]input. "
            "Provide your final answer in plain text. Work autonomously according to your specialty. "
            "Do not ask for clarification. You are one of the following team members: {team_members}."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])

        def parse_tool_calls(response, state: dict) -> dict:
            if not isinstance(response, str):
                response = str(response.content) if hasattr(response, 'content') else str(response)
            tool_pattern = r"\[(\w+)\](.*?)(?=\[\w+\]|$)"
            matches = re.findall(tool_pattern, response, re.DOTALL)
            if matches:
                for tool_name, tool_input in matches:
                    tool_input = tool_input.strip()
                    for tool in tools:
                        if tool.name == tool_name:
                            tool_result = tool.invoke(tool_input)
                            response += f"\nTool Result [{tool_name}]: {tool_result}"
            return {"messages": [HumanMessage(content=response)]}

        return prompt | self.llm | parse_tool_calls

    def agent_node(self, state, agent, name):
        print(f"Agent node '{name}' processing state: {state}")  # Debug agent input
        result = agent.invoke(state)
        if not isinstance(result, dict):
            result = {"messages": [HumanMessage(content=str(result), name=name)]}
        if "messages" in result and not all(isinstance(msg, BaseMessage) for msg in result["messages"]):
            result["messages"] = [HumanMessage(content=str(msg), name=name) for msg in result["messages"]]
        print(f"Agent node '{name}' output: {result}")  # Debug agent output
        return result

    def create_team_supervisor(self, system_prompt: str, members: List[str]) -> Runnable:
        options = ["FINISH"] + members
        system_prompt += (
            f"\nYou are managing a workflow to create a LinkedIn post based on a research paper. "
            f"Given the conversation, determine who should act next or if the task is complete. "
            f"If the task involves gathering information about the paper, select 'Research team' (or 'PaperInformationRetriever' for research tasks). "
            f"If the task involves drafting, editing, or finalizing the LinkedIn post, select 'LinkedIn team' (or 'DocWriter'/'NoteTaker' for writing tasks). "
            f"Only select 'FINISH' when the LinkedIn post has been written, edited, and saved to disk. "
            f"**Return exactly one of: {', '.join(options)}. Output nothing else. No explanations, no plans, no extra text.**"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ]).partial(options=str(options), team_members=", ".join(members))

        def parse_supervisor_output(response: str) -> dict:
            response = response.strip()
            print(f"Supervisor output: {response}")  # Debug supervisor decision
            if response not in options:
                # Check if any message mentions "paper" to route to Research team
                input_messages = [msg.content for msg in prompt.invoke({"messages": []}).messages] if not prompt.invoke({"messages": []}).messages else [state["messages"][-1].content for state in [prompt.input_variables.get("messages", [])]]
                response = "Research team" if any("paper" in msg.lower() for msg in input_messages) else "LinkedIn team"
                if "PaperInformationRetriever" in options and "paper" in input_messages[0].lower():
                    response = "PaperInformationRetriever"
                elif "DocWriter" in options:
                    response = "DocWriter"
            print(f"Parsed supervisor output: {response}")  # Debug parsed decision
            return {"next": response}

        return prompt | self.llm | StrOutputParser() | parse_supervisor_output

    def _create_research_graph(self):
        graph = StateGraph(ResearchTeamState)

        search_agent = self.create_agent(
            [DuckDuckGoSearchRun()],
            "You are a research assistant who can search for up-to-date info using DuckDuckGo search engine."
        )
        search_node = functools.partial(self.agent_node, agent=search_agent, name="Search")

        @tool
        def retrieve_information(query: str) -> str:
            """Retrieve information about the 'Extending Llama-3's Context Ten-Fold Overnight' paper using RAG."""
            if self.rag_pipeline:
                result = self.rag_pipeline.invoke({"question": query})
                print(f"RAG pipeline result for query '{query}': {result}")  # Debug RAG output
                return result
            return "RAG chain not available"

        research_agent = self.create_agent(
            [retrieve_information],
            "You are a research assistant who can provide specific information on the provided paper: 'Extending Llama-3's Context Ten-Fold Overnight'."
        )
        research_node = functools.partial(self.agent_node, agent=research_agent, name="PaperInformationRetriever")

        supervisor_agent = self.create_team_supervisor(
            "You are a supervisor tasked with managing a conversation between the following workers: Search, PaperInformationRetriever.",
            ["Search", "PaperInformationRetriever"]
        )

        graph.add_node("Search", search_node)
        graph.add_node("PaperInformationRetriever", research_node)
        graph.add_node("supervisor", supervisor_agent)

        graph.add_edge("Search", "supervisor")
        graph.add_edge("PaperInformationRetriever", "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"Search": "Search", "PaperInformationRetriever": "PaperInformationRetriever", "FINISH": END}
        )
        graph.set_entry_point("supervisor")
        return graph.compile()

    def _create_doc_writing_graph(self):
        graph = StateGraph(DocWritingState)

        doc_writer_agent = self.create_agent(
            [self.document_tools.write_document, self.document_tools.edit_document, self.document_tools.read_document],
            "You are an expert writing technical LinkedIn posts.\nBelow are files currently in your directory:\n{current_files}"
        )
        context_aware_doc_writer = self.prelude | doc_writer_agent
        doc_writing_node = functools.partial(self.agent_node, agent=context_aware_doc_writer, name="DocWriter")

        note_taking_agent = self.create_agent(
            [self.document_tools.create_outline, self.document_tools.read_document],
            "You are an expert senior researcher tasked with writing a LinkedIn post outline and taking notes to craft a LinkedIn post.\n{current_files}"
        )
        context_aware_note_taking = self.prelude | note_taking_agent
        note_taking_node = functools.partial(self.agent_node, agent=context_aware_note_taking, name="NoteTaker")

        copy_editor_agent = self.create_agent(
            [self.document_tools.write_document, self.document_tools.edit_document, self.document_tools.read_document],
            "You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues\nBelow are files currently in your directory:\n{current_files}"
        )
        context_aware_copy_editor = self.prelude | copy_editor_agent
        copy_editing_node = functools.partial(self.agent_node, agent=context_aware_copy_editor, name="CopyEditor")

        dopeness_editor_agent = self.create_agent(
            [self.document_tools.write_document, self.document_tools.edit_document, self.document_tools.read_document],
            "You are an expert in dopeness, litness, coolness, etc - you edit the document to make sure it's dope. Make sure to use a number of emojis.\nBelow are files currently in your directory:\n{current_files}"
        )
        context_aware_dopeness_editor = self.prelude | dopeness_editor_agent
        dopeness_node = functools.partial(self.agent_node, agent=context_aware_dopeness_editor, name="DopenessEditor")

        supervisor_agent = self.create_team_supervisor(
            "You are a supervisor tasked with managing a conversation between the following workers: {team_members}. You should always verify the technical contents after any edits are made.",
            ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"]
        )

        graph.add_node("DocWriter", doc_writing_node)
        graph.add_node("NoteTaker", note_taking_node)
        graph.add_node("CopyEditor", copy_editing_node)
        graph.add_node("DopenessEditor", dopeness_node)
        graph.add_node("supervisor", supervisor_agent)

        graph.add_edge("DocWriter", "supervisor")
        graph.add_edge("NoteTaker", "supervisor")
        graph.add_edge("CopyEditor", "supervisor")
        graph.add_edge("DopenessEditor", "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"DocWriter": "DocWriter", "NoteTaker": "NoteTaker", "CopyEditor": "CopyEditor", "DopenessEditor": "DopenessEditor", "FINISH": END}
        )
        graph.set_entry_point("supervisor")
        return graph.compile()

    def prelude(self, state):
        written_files = [f.relative_to(self.document_tools.working_directory) for f in self.document_tools.working_directory.rglob("*")]
        if not written_files:
            return {**state, "current_files": "No files written."}
        return {
            **state,
            "current_files": "\nBelow are files your team has written to the directory:\n" + "\n".join([f" - {f}" for f in written_files])
        }

    def _create_supervisor_graph(self):
        class State(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            next: str

        def get_last_message(state: State) -> str:
            print(f"get_last_message input state: {state}")  # Debug state
            return state["messages"][-1].content if state["messages"] else ""

        def join_graph(response: dict):
            print(f"join_graph input response: {response}")  # Debug response
            if not isinstance(response, dict) or "messages" not in response:
                response = {"messages": [HumanMessage(content=str(response))]}
            return {"messages": response["messages"]}

        graph = StateGraph(State)

        graph.add_node("Research team", get_last_message | self.research_graph | join_graph)
        graph.add_node("LinkedIn team", get_last_message | self.doc_writing_graph | join_graph)

        supervisor_node = self.create_team_supervisor(
            "You are a supervisor tasked with managing a conversation between the following teams: {team_members}.",
            ["Research team", "LinkedIn team"]
        )
        graph.add_node("supervisor", supervisor_node)

        graph.add_edge("Research team", "supervisor")
        graph.add_edge("LinkedIn team", "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"Research team": "Research team", "LinkedIn team": "LinkedIn team", "FINISH": END}
        )
        graph.set_entry_point("supervisor")
        return graph.compile()

    def invoke(self, message: str):
        try:
            print(f"Invoking supervisor graph with message: {message}")  # Debug initial message
            for s in self.supervisor_graph.stream(
                {"messages": [HumanMessage(content=message)]},
                {"recursion_limit": 30}
            ):
                if "__end__" not in s:
                    print(s)
                    print("---")
        except Exception as e:
            print(f"Error testing full system: {e}")
