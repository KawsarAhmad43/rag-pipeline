import re
from langchain.tools import Tool
from typing import Dict, Any
from datetime import datetime

class WebSearchManager:
    """Manager for web search functionality"""

    @staticmethod
    def web_search(query: str) -> str:
        """Perform web search using Tavily API directly"""
        try:
            from config.settings import settings
            if not settings.TAVILY_API_KEY:
                return "❌ Tavily API key not configured"

            # Try to use langchain-tavily
            try:
                from langchain_tavily import TavilySearch

                tavily_tool = TavilySearch(
                    max_results=5,
                    topic="general",
                    include_answer=True,
                    include_raw_content=False,
                    search_depth="advanced",
                    api_key=settings.TAVILY_API_KEY
                )

                results = tavily_tool.invoke({"query": query})
                return f"**Web Search Results for: {query}**\n\n{results}"

            except ImportError:
                # Fallback to requests if langchain-tavily not available
                import requests

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.TAVILY_API_KEY}"
                }

                data = {
                    "query": query,
                    "max_results": 5,
                    "include_answer": True,
                    "include_raw_content": False,
                    "search_depth": "advanced"
                }

                response = requests.post(
                    "https://api.tavily.com/search",
                    headers=headers,
                    json=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Format the results
                    formatted_results = f"**Web Search Results for: {query}**\n\n"

                    if result.get('answer'):
                        formatted_results += f"**Summary:** {result['answer']}\n\n"

                    for idx, r in enumerate(result.get('results', []), 1):
                        formatted_results += f"{idx}. **{r.get('title', 'No title')}**\n"
                        formatted_results += f"   {r.get('content', 'No content')}\n"
                        formatted_results += f"   Source: {r.get('url', 'No URL')}\n\n"

                    return formatted_results
                else:
                    return f"❌ Search failed with status: {response.status_code}"

        except Exception as e:
            return f"❌ Search error: {str(e)}"

class UserInfoManager:
    """Simple user information manager"""

    def __init__(self):
        self.user_info_storage = {}

    def get_user_info(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.user_info_storage:
            self.user_info_storage[session_id] = {
                "name": None,
                "email": None,
                "first_interaction": True
            }
        return self.user_info_storage[session_id]

    def extract_user_info(self, text: str, session_id: str) -> str:
        user_info = self.get_user_info(session_id)

        name_match = re.search(r"(?:my name is|i am|name is|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.IGNORECASE)
        email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)

        response = []

        if name_match:
            user_info["name"] = name_match.group(1).title()
            response.append(f"✅ I've noted your name: {user_info['name']}")

        if email_match:
            user_info["email"] = email_match.group(0)
            response.append(f"✅ I've noted your email: {user_info['email']}")

        return " ".join(response) if response else ""

    def clear_user_info(self, session_id: str):
        if session_id in self.user_info_storage:
            del self.user_info_storage[session_id]

class ToolsManager:
    """Simple tools manager for basic functionality"""

    def __init__(self):
        self.user_info_manager = UserInfoManager()

    def create_tools_for_session(self, session_id: str) -> list:
        """Create basic tools for a session"""

        def extract_info_with_session(text: str) -> str:
            return self.user_info_manager.extract_user_info(text, session_id)

        def handle_first_with_session(query: str) -> str:
            return self._handle_first_interaction(session_id)

        def get_name_with_session(query: str) -> str:
            user_info = self.user_info_manager.get_user_info(session_id)
            name = user_info.get("name")
            return f"Your name is {name}" if name else "I don't have your name stored yet."

        def get_email_with_session(query: str) -> str:
            user_info = self.user_info_manager.get_user_info(session_id)
            email = user_info.get("email")
            return f"Your email is {email}" if email else "I don't have your email stored yet."

        # Simple tool set - no web search tool here since it's handled by the agent
        tools = [
            Tool(
                name="extract_user_info",
                func=extract_info_with_session,
                description="Use this if the user provides their name or email."
            ),
            Tool(
                name="get_user_name",
                func=get_name_with_session,
                description="Use this when the user asks about their name."
            ),
            Tool(
                name="get_user_email",
                func=get_email_with_session,
                description="Use this when the user asks about their email."
            ),
            Tool(
                name="handle_greeting",
                func=handle_first_with_session,
                description="Use this for first interactions or greetings."
            )
        ]

        return tools

    def _handle_first_interaction(self, session_id: str) -> str:
        user_info = self.user_info_manager.get_user_info(session_id)

        if user_info["first_interaction"]:
            user_info["first_interaction"] = False
            return "👋 Hello! I'm your AI assistant. I can help with questions, analyze documents if you upload them, and search the web if you say 'search on web' or 'web search'. What would you like to know?"

        return ""

    def get_user_info_for_session(self, session_id: str) -> Dict[str, Any]:
        return self.user_info_manager.get_user_info(session_id)

    def clear_user_info(self, session_id: str):
        self.user_info_manager.clear_user_info(session_id)
