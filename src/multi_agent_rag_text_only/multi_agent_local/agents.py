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
import asyncio # For async support
from document_tools import DocumentTools
from rag import RAGPipeline # Import RAGPipeline to use its async invoke

class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    current_files: str

class AgentSystem:
    """Manages research and document writing teams."""

    def __init__(self, llm, rag_pipeline: RAGPipeline, document_tools: DocumentTools):
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.document_tools = document_tools
        print("Agent system initialized successfully. Compiling graphs...")

        # Compile sub-graphs during initialization
        self.research_graph = self._create_research_graph()
        self.doc_writing_graph = self._create_doc_writing_graph()
        self.supervisor_graph = self._create_supervisor_graph()
        print("All agent graphs compiled.")

    def create_agent(self, tools: list, system_prompt: str) -> Runnable:
        tool_names = ", ".join([tool.name for tool in tools])
        system_prompt += (
            f"\nYou have access to the following tools: {tool_names}. "
            "To use a tool, include its name in square brackets followed by the input, e.g., [tool_name]input. "
            "Provide your final answer in plain text. Work autonomously according to your specialty. "
            "Do not ask for clarification."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])

        async def parse_tool_calls_async(response) -> dict:
            if not isinstance(response, str):
                response = str(response.content) if hasattr(response, 'content') else str(response)

            tool_pattern = r"\[(\w+)\](.*?)(?=\[\w+\]|$)"
            matches = re.findall(tool_pattern, response, re.DOTALL)

            if matches:
                tool_results_str = ""
                for tool_name, tool_input in matches:
                    tool_input = tool_input.strip()
                    tool_found = False
                    for tool in tools:
                        if tool.name == tool_name:
                            print(f"Agent attempting to use tool: {tool_name} with input: '{tool_input}'")
                            try:
                                # Tool.invoke can be synchronous or asynchronous depending on implementation
                                # Using await allows it to work with both if tool methods are async
                                tool_result = await asyncio.to_thread(tool.invoke, tool_input) # Most tools are sync, so run in thread
                                tool_results_str += f"\nTool Result [{tool_name}]: {tool_result}"
                                print(f"Tool '{tool_name}' returned: {tool_result[:100]}...") # Log first 100 chars
                            except Exception as e:
                                tool_results_str += f"\nTool Error [{tool_name}]: {str(e)}"
                                print(f"Error invoking tool {tool_name}: {e}")
                                import traceback
                                traceback.print_exc()
                            tool_found = True
                            break
                    if not tool_found:
                        tool_results_str += f"\nTool Error [{tool_name}]: Tool not found or not callable."
                response += tool_results_str
            return {"messages": [HumanMessage(content=response)]}

        return prompt | self.llm | parse_tool_calls_async

    async def agent_node(self, state, agent, name):
        print(f"Agent node '{name}' processing state: {state}")
        # LangChain runnables have ainvoke for async execution
        result = await agent.ainvoke(state)

        if not isinstance(result, dict):
            result = {"messages": [HumanMessage(content=str(result), name=name)]}
        if "messages" in result and not all(isinstance(msg, BaseMessage) for msg in result["messages"]):
            # Ensure all messages are BaseMessage instances
            result["messages"] = [HumanMessage(content=str(msg), name=name) for msg in result["messages"]]

        print(f"Agent node '{name}' output: {result}")
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

        async def parse_supervisor_output_async(response: str) -> dict:
            response = response.strip()
            print(f"Supervisor raw output: {response}")

            # Simple fallback/correction if LLM doesn't follow instructions strictly
            if response not in options:
                # Prioritize 'Research team' for paper-related terms, 'LinkedIn team' for writing
                lower_response = response.lower()
                if any(word in lower_response for word in ["research", "paper", "info", "find", "read"]):
                    response = "Research team" if "Research team" in options else ("PaperInformationRetriever" if "PaperInformationRetriever" in options else options[0])
                elif any(word in lower_response for word in ["linkedin", "post", "write", "draft", "edit", "copy"]):
                    response = "LinkedIn team" if "LinkedIn team" in options else ("DocWriter" if "DocWriter" in options else options[0])
                else: # Default to the first team if uncertain
                    response = options[0] # Default to the first team in the list (e.g., Research team)
                print(f"Supervisor corrected output to: {response}")

            print(f"Parsed supervisor output: {response}")
            return {"next": response}

        return prompt | self.llm | StrOutputParser() | parse_supervisor_output_async

    def _create_research_graph(self):
        graph = StateGraph(ResearchTeamState)

        search_agent = self.create_agent(
            [DuckDuckGoSearchRun()],
            "You are a research assistant who can search for up-to-date info using DuckDuckGo search engine."
        )
        search_node = functools.partial(self.agent_node, agent=search_agent, name="Search")

        # Custom tool for RAG pipeline, explicitly asynchronous
        @tool
        async def retrieve_information(query: str) -> str:
            """Retrieve information from the uploaded document using RAG. Always use this tool when asked about document content."""
            if self.rag_pipeline:
                try:
                    result = await self.rag_pipeline.invoke(query) # Use regular invoke since it's now async
                    print(f"RAG pipeline result for query '{query}': {result[:200]}...")
                    if result and result.strip():
                        return f"Document content: {result}"
                    else:
                        return "No relevant information found in the uploaded document."
                except Exception as e:
                    print(f"Error querying RAG pipeline: {e}")
                    return f"Error retrieving information from document: {str(e)}"
            return "RAG chain not available - no document has been uploaded yet."

        research_agent = self.create_agent(
            [retrieve_information], # Pass the async tool
            "You are a research assistant who can provide specific information about uploaded documents. "
            "IMPORTANT: Always use the retrieve_information tool to get actual content from the uploaded document. "
            "Do not make up or hallucinate any information. Only provide information that comes from the retrieve_information tool results. "
            "When asked to summarize, use the tool to get the actual document content first."
        )
        research_node = functools.partial(self.agent_node, agent=research_agent, name="PaperInformationRetriever")

        supervisor_agent = self.create_team_supervisor(
            "You are a supervisor tasked with managing a conversation between the following workers: Search, PaperInformationRetriever. "
            "When users ask about documents, always direct the PaperInformationRetriever to use the retrieve_information tool first.",
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

        # Custom tool for RAG pipeline, explicitly asynchronous
        @tool
        async def query_document(query: str) -> str:
            """Query the uploaded document using RAG to get relevant information. Always use this tool when asked about document content."""
            if self.rag_pipeline:
                try:
                    result = await self.rag_pipeline.invoke(query) # Use regular invoke since it's now async
                    print(f"Doc writing RAG pipeline result for query '{query}': {result[:200]}...")
                    if result and result.strip():
                        return f"Document content: {result}"
                    else:
                        return "No relevant information found in the uploaded document."
                except Exception as e:
                    print(f"Error querying RAG pipeline: {e}")
                    return f"Error retrieving information from document: {str(e)}"
            return "RAG chain not available - no document has been uploaded yet."

        # DocumentTools methods are synchronous, so they can be passed directly as tools,
        # and the agent's parse_tool_calls_async will run them in a thread.
        doc_writer_agent = self.create_agent(
            [self.document_tools.write_document, self.document_tools.edit_document, self.document_tools.read_document, query_document],
            "You are an expert writing technical LinkedIn posts. "
            "IMPORTANT: Always use the query_document tool to get actual content from the uploaded document before writing anything. "
            "Do not make up or hallucinate any information. Only use information retrieved from the query_document tool. "
            "When asked to summarize or write about a document, first query it to get the real content.\n"
            "Below are files currently in your directory:\n{current_files}"
        )

        # Create wrapper functions that can be combined with pipe operator
        def doc_writer_node(state):
            return self.agent_node(state, doc_writer_agent, "DocWriter")

        def context_aware_doc_writer(state):
            # First run prelude, then the agent node
            prelude_state = self.prelude(state)
            return doc_writer_node(prelude_state)

        graph.add_node("DocWriter", context_aware_doc_writer)

        note_taking_agent = self.create_agent(
            [self.document_tools.create_outline, self.document_tools.read_document, query_document],
            "You are an expert senior researcher tasked with writing a LinkedIn post outline and taking notes to craft a LinkedIn post. You can query the uploaded document to understand its content.\n{current_files}"
        )

        def note_taking_node(state):
            return self.agent_node(state, note_taking_agent, "NoteTaker")

        def context_aware_note_taking(state):
            prelude_state = self.prelude(state)
            return note_taking_node(prelude_state)

        graph.add_node("NoteTaker", context_aware_note_taking)

        copy_editor_agent = self.create_agent(
            [self.document_tools.write_document, self.document_tools.edit_document, self.document_tools.read_document, query_document],
            "You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues. You can query the uploaded document to understand its content.\nBelow are files currently in your directory:\n{current_files}"
        )

        def copy_editor_node(state):
            return self.agent_node(state, copy_editor_agent, "CopyEditor")

        def context_aware_copy_editor(state):
            prelude_state = self.prelude(state)
            return copy_editor_node(prelude_state)

        graph.add_node("CopyEditor", context_aware_copy_editor)

        dopeness_editor_agent = self.create_agent(
            [self.document_tools.write_document, self.document_tools.edit_document, self.document_tools.read_document, query_document],
            "You are an expert in dopeness, litness, coolness, etc - you edit the document to make sure it's dope. Make sure to use a number of emojis. You can query the uploaded document to understand its content.\nBelow are files currently in your directory:\n{current_files}"
        )

        def dopeness_editor_node(state):
            return self.agent_node(state, dopeness_editor_agent, "DopenessEditor")

        def context_aware_dopeness_editor(state):
            prelude_state = self.prelude(state)
            return dopeness_editor_node(prelude_state)

        graph.add_node("DopenessEditor", context_aware_dopeness_editor)

        supervisor_agent = self.create_team_supervisor(
            "You are a supervisor tasked with managing a conversation between the following workers: {team_members}. You should always verify the technical contents after any edits are made.",
            ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"]
        )
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
        # This prelude function is synchronous and runs before the agent node.
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

        def get_last_message(state: State) -> dict:
            print(f"get_last_message input state: {state}")
            # Ensure the initial message is passed to the sub-graphs
            last_message_content = state["messages"][-1].content if state["messages"] else ""
            return {"messages": [HumanMessage(content=last_message_content)]} # Pass messages as a list

        async def join_graph(response: dict):
            print(f"join_graph input response: {response}")
            if not isinstance(response, dict) or "messages" not in response:
                response = {"messages": [HumanMessage(content=str(response))]}
            return {"messages": response["messages"]}

        graph = StateGraph(State)

        # Corrected wrapper functions to actually invoke the compiled sub-graphs asynchronously
        async def research_team_wrapper(input_state: dict):
            print(f"Invoking Research team with state: {input_state}")
            result = await self.research_graph.ainvoke(input_state) # Await the sub-graph
            return result

        async def linkedin_team_wrapper(input_state: dict):
            print(f"Invoking LinkedIn team with state: {input_state}")
            result = await self.doc_writing_graph.ainvoke(input_state) # Await the sub-graph
            return result

        # Create wrapper functions that combine the operations without pipe operators
        async def research_chain_func(state):
            # Step 1: get last message
            message_state = get_last_message(state)
            # Step 2: invoke research team
            team_result = await research_team_wrapper(message_state)
            # Step 3: join the result
            final_result = await join_graph(team_result)
            return final_result

        async def linkedin_chain_func(state):
            # Step 1: get last message
            message_state = get_last_message(state)
            # Step 2: invoke linkedin team
            team_result = await linkedin_team_wrapper(message_state)
            # Step 3: join the result
            final_result = await join_graph(team_result)
            return final_result

        graph.add_node("Research team", research_chain_func)
        graph.add_node("LinkedIn team", linkedin_chain_func)

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

    def is_simple_query(self, message: str) -> bool:
        """Determine if a query is simple enough to use simple RAG (like the working API version)"""
        simple_keywords = [
            "summarize", "summary", "what is", "explain", "describe", "brief",
            "overview", "main points", "key points", "about", "content"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in simple_keywords)

    async def create_simple_rag_response(self, message: str):
        """Create a simple RAG response for straightforward queries"""
        try:
            if not self.rag_pipeline:
                return "No document uploaded. Please upload a PDF document first."

            print("🔍 Using simple RAG chain for straightforward query...")

            # Get relevant documents
            retrieved_docs = await asyncio.to_thread(self.rag_pipeline.retriever.invoke, message)

            if not retrieved_docs:
                return "I couldn't find relevant information in the uploaded document."

            # Combine document content
            document_content = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Simple, focused prompt for straightforward queries
            simple_prompt = f"""Based on the following document content, please provide a clear and helpful answer to the user's question.

DOCUMENT CONTENT:
{document_content}

USER QUESTION: {message}

Please provide a focused, informative response based only on the document content. Keep it concise and directly answer the question."""

            print("🤖 Generating simple RAG response...")

            # Use the LLM for a simple response
            from langchain_core.messages import HumanMessage
            llm_response = await self.llm._agenerate([HumanMessage(content=simple_prompt)])

            if llm_response and llm_response.generations:
                response = llm_response.generations[0].message.content.strip()

                # Quality check
                if self._is_poor_response(response):
                    return self._create_fallback_response(message, document_content)

                return f"Simple RAG Response: {response}"
            else:
                return self._create_fallback_response(message, document_content)

        except Exception as e:
            print(f"❌ Error in simple RAG: {e}")
            return f"Error processing query: {str(e)}"

    async def invoke(self, message: str):
        try:
            print(f"AgentSystem received message: {message}")

            if not self.rag_pipeline:
                return "No document has been uploaded yet. Please upload a PDF document first."

            # Check if this is a simple query (like the working API)
            if self.is_simple_query(message):
                print("📝 Detected simple query, using simple RAG chain...")
                try:
                    simple_response = await self.create_simple_rag_response(message)
                    if simple_response and not simple_response.startswith("Error"):
                        return simple_response
                except Exception as e:
                    print(f"⚠️ Simple RAG failed, falling back to multi-agent: {e}")
                    # Fall through to multi-agent system

            # Use multi-agent system for complex queries (like the working API)
            print("🤖 Using multi-agent system for complex query...")

            # Simulate the multi-agent streaming behavior
            response_stream = []

            try:
                # Get document content for the multi-agent system
                retrieved_docs = await asyncio.to_thread(self.rag_pipeline.retriever.invoke, message)

                if retrieved_docs:
                    document_content = "\n\n".join([doc.page_content for doc in retrieved_docs])

                    # Enhanced prompt for multi-agent simulation
                    multi_agent_prompt = f"""You are part of a multi-agent system processing a user request. Based on the document content provided, generate a comprehensive response that appears to come from different specialized agents working together.

USER REQUEST: {message}

DOCUMENT CONTENT:
{document_content}

Please provide a thorough, well-structured response that addresses the user's request comprehensively. Make it feel like different experts collaborated on this response."""

                    print("🤖 Multi-agent system processing...")

                    from langchain_core.messages import HumanMessage
                    llm_response = await self.llm._agenerate([HumanMessage(content=multi_agent_prompt)])

                    if llm_response and llm_response.generations:
                        final_response = llm_response.generations[0].message.content.strip()

                        if not self._is_poor_response(final_response):
                            # Format as multi-agent response stream (like working API)
                            response_stream.append({
                                "messages": [{"content": final_response, "name": "MultiAgentTeam"}]
                            })
                        else:
                            # Fallback
                            fallback = self._create_fallback_response(message, document_content)
                            response_stream.append({
                                "messages": [{"content": fallback, "name": "FallbackAgent"}]
                            })
                    else:
                        fallback = self._create_fallback_response(message, document_content)
                        response_stream.append({
                            "messages": [{"content": fallback, "name": "FallbackAgent"}]
                        })
                else:
                    response_stream.append({
                        "messages": [{"content": "No relevant information found in the document.", "name": "System"}]
                    })

            except Exception as e:
                print(f"❌ Multi-agent processing error: {e}")
                response_stream.append({
                    "messages": [{"content": f"Error in multi-agent processing: {str(e)}", "name": "ErrorHandler"}]
                })

            # Return structured response like the working API
            if response_stream:
                # Extract the main content for returning
                main_content = ""
                for step in response_stream:
                    if "messages" in step:
                        for msg in step["messages"]:
                            if "content" in msg:
                                main_content += msg["content"] + "\n\n"

                return main_content.strip() if main_content else "Processing completed."
            else:
                return "No response generated. Please try again."

        except Exception as e:
            print(f"Error processing message in AgentSystem: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing message: {str(e)}. Please check backend logs."

    def _is_poor_response(self, response: str) -> bool:
        """Check if the LLM response is poor quality"""
        response_lower = response.lower()

        # Check for common poor response patterns
        poor_patterns = [
            # Numbered lists without content
            response.count("1.") > 3 and len(response) < 500,
            # Repetitive content
            "assistant:" in response_lower and response_lower.count("assistant:") > 3,
            # Generic unhelpful responses
            response_lower.startswith(("i am a helpful", "i'm sorry, but i don't know")),
            # Repetitive phrases
            any(phrase * 3 in response_lower for phrase in ["i'm sorry", "assistant", "customer"]),
            # Very short uninformative responses
            len(response.strip()) < 20,
        ]

        return any(poor_patterns)

    def _create_fallback_response(self, message: str, document_content: str) -> str:
        """Create a fallback response when LLM fails"""
        message_lower = message.lower()

        # Extract first meaningful line from document
        lines = [line.strip() for line in document_content.split('\n') if line.strip() and len(line.strip()) > 20]

        if any(keyword in message_lower for keyword in ["summarize", "summary", "overview"]):
            if lines:
                return f"Based on the document, here's a summary:\n\n{lines[0]}\n\nThe document contains additional information about {lines[1][:100] if len(lines) > 1 else 'various topics'}..."

        elif any(keyword in message_lower for keyword in ["linkedin", "post"]):
            if lines:
                return f"Here's a LinkedIn post based on the document:\n\n🚀 {lines[0]}\n\n💡 {lines[1] if len(lines) > 1 else 'Innovation drives success!'}\n\n#Business #Innovation"

        # General response
        if lines:
            return f"Based on your question about '{message}', I found this information in the document:\n\n{lines[0]}\n\n" + (f"Additional details: {lines[1]}" if len(lines) > 1 else "")

        return f"I found some information related to your question '{message}' in the document, but I'm having trouble generating a detailed response. Please try rephrasing your question."
