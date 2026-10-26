from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from duckduckgo_search import DDGS
import os
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime
import re

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="DuckDuckGo and Gemini Web Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Retrieve Google API Key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Configure Gemini API with the retrieved key
genai.configure(api_key=google_api_key)

# Request model
class SearchRequest(BaseModel):
    query: str
    model: Optional[str] = "gemini-1.5-flash"  # Note: Using gemini-1.5-flash as gemini-2.5-flash may not be valid
    max_results: Optional[int] = 5

@app.get("/")
async def root():
    return {"message": "Welcome to DuckDuckGo and Gemini Web Search API"}

@app.post("/search")
async def search_web(request: SearchRequest):
    try:
        # Check if the query is asking for the current date or time
        date_time_pattern = re.compile(r"(current|today|now)\s+(date|time)", re.IGNORECASE)
        is_date_query = bool(date_time_pattern.search(request.query))

        if is_date_query:
            # Use system date/time for date-related queries
            current_date = datetime.now().strftime("%B %d, %Y")
            context = f"The current date is {current_date}."
            search_results = []  # No need for search results
        else:
            # Perform DuckDuckGo search
            ddg = DDGS()
            search_results = ddg.text(keywords=request.query, max_results=request.max_results)

            # Format search results as context
            context = "\n".join(
                f"[{result['title']}]({result['href']}): {result['body']}"
                for result in search_results
            ) if search_results else "No search results found."

        # Initialize the Gemini model
        model = genai.GenerativeModel(model_name=request.model)

        # Create prompt with search results as context
        prompt = f"""
        You are an AI assistant that provides accurate and concise answers based on web search results or provided context.
        Use the following context to answer the query: "{request.query}"

        **Context:**
        {context}

        Provide a clear and concise response to the query. Include citations to the search results where applicable.
        If no relevant information is found or the context is insufficient, state that clearly and provide the best possible answer.
        For date or time-related queries, prioritize the provided context if it includes a specific date or time.
        """

        # Generate content with Gemini (synchronous call)
        response = model.generate_content(prompt)

        # Extract the text from response
        result = response.text if hasattr(response, 'text') else str(response)

        return {
            "query": request.query,
            "response": result,
            "search_results": search_results,
            "model": request.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
