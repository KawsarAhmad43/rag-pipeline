from utils.vector_search import vector_search
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import torch
import logging
import gc
import numpy as np

from utils.logging import logger
from app.embedding.embedding import get_embedding_model

def search_documents(query, context_type="pdf", k=5):
    """Search for relevant documents from the Simple RAG knowledge base."""
    try:
        from app.embedding.embedding import get_embedding_model
        embedding_model = get_embedding_model()
        embeddings = embedding_model.embed_query(query)
        results = vector_search.search(embeddings, context_type=context_type)

        logger.debug(f"Found {len(results)} documents for query: {query[:50]}... from Simple RAG knowledge base")
        return results[:k]
    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        return []

def create_context_from_docs(documents):
    """Create a structured and professional context string from retrieved documents."""
    if not documents:
        return "No specific document context found for this query."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        metadata = getattr(doc, 'metadata', {})
        source = metadata.get("source", "Uploaded Document")
        page = metadata.get("page", "Unknown")

        context_parts.append(
            f"Segment {i} (Source: {source}, Page: {page}):\n{content.strip()}\n---"
        )
    return "\n".join(context_parts)

def get_agent_response(query: str, context: str = None) -> str:
    """Generate response using RAG approach with simplified prompt and robust error handling."""
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        return "Please provide a valid question."

    query = query.strip()
    logger.debug(f"Processing query: {query[:50]}...")

    # Handle simple greetings
    if query.lower() in ["hello", "hi", "hey"]:
        return "Hi! How can I assist you?"

    context = context if context else "No context provided."

    has_meaningful_context = context and context != "No context provided." and context != "No relevant PDF content found." and context != "No relevant information found in the knowledge base." and context != "No specific document context found for this query."

    # Define a professional prompt template for PDF context (FLEXIBLE & HELPFUL)
    # Following Agentic RAG's successful approach
    pdf_prompt_template = PromptTemplate.from_template(
        """
        You are a highly skilled AI assistant. Your goal is to provide exceptional, accurate, and helpful responses.

        [SYSTEM INFO]
        Document Loaded: Yes

        [GUIDELINES]
        1. **Prioritize Document Context**: If [DOCUMENT CONTEXT] is provided, use it as your primary source of truth. Always cite sources (document name, page number) when using document information.
        2. **Global Queries**: If the user asks for a summary or overview, synthesize all provided context segments into a comprehensive response.
        3. **Conversational Fallback**: If the document context is missing or doesn't fully answer the question, use your pre-trained knowledge to remain helpful. Be honest and mention that specific information isn't in the uploaded document, but provide general assistance anyway. Do NOT give a hard refusal.
        4. **Professional Style**: Use professional language and Markdown formatting (bolding, bullet points, lists) for readability.

        [CONVERSATION HISTORY]
        {history}

        [DOCUMENT CONTEXT]
        {retrieved_info}

        [USER QUESTION]
        {user_input}

        Assistant Response:
        """
    )

    # Define a sophisticated prompt template for general queries
    general_prompt_template = PromptTemplate.from_template(
        """
        You are a sophisticated and helpful AI assistant. Provide a direct, factual, and well-structured answer to the user's question.

        Instructions:
        1. Accuracy First: Use your general knowledge to provide the most accurate and up-to-date response possible.
        2. Conciseness: Be direct and avoid unnecessary fluff while remaining conversational.
        3. Formatting: Use Markdown for clarity (e.g., bullet points for lists).

        [Conversation History]
        {history}

        [User Question]
        {user_input}

        Assistant Response:
        """
    )

    from app.llm.llm import get_llm
    from utils.database import db
    llm_pipeline_wrapper = get_llm()

    # Check if this is a local pipeline or an API instance
    hf_pipeline = getattr(llm_pipeline_wrapper, "pipeline", None)

    if hf_pipeline:
        tokenizer = hf_pipeline.tokenizer
    else:
        # For OpenAI/Gemini providers, we don't have a direct pipeline attribute
        tokenizer = None

    try:
        # Get conversation history from chat memory
        history = db.search_chat_memory(query, get_embedding_model(), context_type="pdf" if has_meaningful_context else "not_pdf")
        history = history if history else "No previous conversation history."

        # Select the appropriate prompt template
        prompt_template = pdf_prompt_template if has_meaningful_context else general_prompt_template

        # Format the prompt
        prompt = prompt_template.format(
            history=history,
            retrieved_info=context if has_meaningful_context else "",  # Empty for general prompt
            user_input=query
        )
        logger.debug(f"Final prompt sent to LLM:\n---START PROMPT---\n{prompt}\n---END PROMPT---")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                with torch.no_grad():
                    raw_response = llm_pipeline_wrapper.invoke(prompt)
                    logger.debug(f"Raw LLM output: '{raw_response}'")

                if raw_response:
                    response = str(raw_response).strip()

                    # Extract text after "Answer:"
                    answer_split = response.split("Answer:")
                    if len(answer_split) > 1:
                        response = answer_split[-1].strip()
                    else:
                        # Fallback: remove prompt text
                        response = response.replace(prompt, "").strip()

                    # Check for refusal phrases
                    refusal_phrases = [
                        "i cannot answer this question",
                        "i am sorry, but i don't have enough information",
                        "i couldn't generate a proper response",
                        "the llm produced no output"
                    ]

                    lower_response = response.lower()

                    if any(phrase in lower_response for phrase in refusal_phrases):
                        # Instead of a hard refusal, we return the response anyway but log a warning
                        # or provide a more gentle fallback if it was truly empty
                        if not response or response.strip() == "":
                            return "I couldn't find a specific answer in the documents, and I'm having trouble generating a general response. Could you try rephrasing?"
                        return response


                    if len(response) > 1000:
                        response = response[:1000] + "..."

                    if not response or response.strip() == "":
                        if attempt < max_retries - 1:
                            logger.warning(f"Empty response on attempt {attempt + 1}, retrying...")
                            continue
                        return "I couldn't generate a response. Please try again with a different query."

                    return response
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"No response on attempt {attempt + 1}, retrying...")
                        continue
                    return "I couldn't generate a response. Please try again with a different query."

            except Exception as e:
                logger.error(f"Error generating response on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                return "I couldn't generate a response due to an error. Please try again."

    finally:
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return "I couldn't generate a response after multiple attempts. Please try again."


def get_pdf_analyzer_response(query: str, documents) -> str:
    """
    Advanced PDF analysis response with detailed insights when PDF Analyzer tool is enabled.
    Provides comprehensive summaries, key points extraction, and structured analysis.
    """
    if not query or not documents:
        return "No documents available for analysis."

    query = query.strip()
    logger.info(f"📊 PDF Analyzer: Processing advanced analysis for query: {query[:50]}...")

    # Build comprehensive context from all documents
    full_context = ""
    sources_info = []

    for i, doc in enumerate(documents, 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        metadata = getattr(doc, 'metadata', {})
        source = metadata.get("source", "Unknown Document")
        page = metadata.get("page", "Unknown")

        full_context += f"\n--- SOURCE {i} ---\n"
        full_context += f"Document: {source} | Page: {page}\n"
        full_context += f"{content}\n"

        sources_info.append(f"[{source}, Page {page}]")

    # Advanced analysis prompt
    pdf_analyzer_prompt = PromptTemplate.from_template(
        """
        You are an advanced PDF Analysis AI assistant with expertise in document comprehension and information extraction.

        Your task is to provide a comprehensive, structured analysis of the provided PDF content to answer the user's query.

        **Analysis Guidelines:**
        1. **Detailed Answer**: Provide a thorough, well-structured answer to the user's question
        2. **Key Points**: Extract and list the most important points related to the query
        3. **Source Attribution**: Always cite specific sources (document name and page number)
        4. **Structured Format**: Use clear sections with headers, bullet points, and formatting
        5. **Comprehensive Coverage**: Include all relevant information from the documents
        6. **Context Awareness**: Synthesize information across multiple sources if available

        **Available Document Sources:**
        {sources_list}

        **Document Content:**
        {document_context}

        **User Query:**
        {user_query}

        **Analysis Response:**
        Provide your comprehensive analysis below with the following structure:

        ## Answer
        [Detailed answer to the query]

        ## Key Points
        - [Key point 1]
        - [Key point 2]
        - [Additional points as needed]

        ## Sources Referenced
        [List of sources used with page numbers]

        Assistant Analysis:
        """
    )

    from app.llm.llm import get_llm
    llm_pipeline_wrapper = get_llm()

    try:
        # Format the specialized analysis prompt
        prompt = pdf_analyzer_prompt.format(
            sources_list="\n".join(sources_info),
            document_context=full_context,
            user_query=query
        )

        logger.debug(f"PDF Analyzer prompt length: {len(prompt)} characters")

        with torch.no_grad():
            raw_response = llm_pipeline_wrapper.invoke(prompt)
            logger.debug(f"PDF Analyzer raw output: {raw_response[:200]}...")

            if raw_response:
                response = str(raw_response).strip()

                # Clean up response
                if "Analysis Response:" in response:
                    response = response.split("Analysis Response:")[-1].strip()
                elif "Assistant Analysis:" in response:
                    response = response.split("Assistant Analysis:")[-1].strip()

                logger.info(f"✅ PDF Analyzer: Generated analysis ({len(response)} chars)")
                return response
            else:
                logger.warning("PDF Analyzer: Empty response from LLM")
                return "Unable to generate PDF analysis. Please try again."

    except Exception as e:
        logger.error(f"Error in PDF Analyzer: {str(e)}")
        return f"Error generating PDF analysis: {str(e)}"

    finally:
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def get_web_search_response(query: str, pdf_context: str = None) -> str:
    """
    Generate response using web search results.
    Fetches information from the web and combines it with PDF context if available.
    """
    if not query:
        return "Please provide a valid query."

    query = query.strip()
    logger.info(f"🌐 Web Search: Searching for: {query[:50]}...")

    try:
        # Use DuckDuckGo for web search
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            # Get top 5 search results
            results = list(ddgs.text(query, max_results=5))

            if not results:
                logger.warning("Web Search: No results found")
                return "I couldn't find any web results for your query. Please try rephrasing."

            # Build context from web search results
            web_context = "**Web Search Results:**\n\n"
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                snippet = result.get('body', 'No description')
                url = result.get('href', '')

                web_context += f"**Result {i}:** {title}\n"
                web_context += f"Source: {url}\n"
                web_context += f"{snippet}\n\n"

            logger.info(f"✅ Web Search: Found {len(results)} results")

            # Create prompt combining web results and optional PDF context
            if pdf_context and pdf_context != "No context provided." and pdf_context != "No relevant PDF content found.":
                combined_prompt = PromptTemplate.from_template(
                    """
                    You are a helpful AI assistant with access to both web search results and document context.

                    **User Query:**
                    {user_query}

                    **Web Search Information:**
                    {web_results}

                    **Document Context (if relevant):**
                    {doc_context}

                    **Instructions:**
                    1. Synthesize information from both web results and document context
                    2. Provide a comprehensive, accurate answer
                    3. Cite sources when possible (mention "web sources" or "document")
                    4. Use clear, professional language
                    5. If web results provide newer/different information than the document, mention both perspectives

                    **Response:**
                    """
                )

                prompt = combined_prompt.format(
                    user_query=query,
                    web_results=web_context,
                    doc_context=pdf_context
                )
                logger.info("🔄 Combining web search results with PDF context")
            else:
                web_only_prompt = PromptTemplate.from_template(
                    """
                    You are a helpful AI assistant with access to web search results.

                    **User Query:**
                    {user_query}

                    **Web Search Information:**
                    {web_results}

                    **Instructions:**
                    1. Synthesize information from the web search results
                    2. Provide a clear, accurate, and comprehensive answer
                    3. Mention "according to web sources" when citing information
                    4. Use professional, conversational language
                    5. If results are conflicting, present different perspectives

                    **Response:**
                    """
                )

                prompt = web_only_prompt.format(
                    user_query=query,
                    web_results=web_context
                )
                logger.info("🌐 Using web search results only")

            # Generate response
            from app.llm.llm import get_llm
            llm_pipeline_wrapper = get_llm()

            with torch.no_grad():
                raw_response = llm_pipeline_wrapper.invoke(prompt)
                logger.debug(f"Web Search raw output: {raw_response[:200]}...")

                if raw_response:
                    response = str(raw_response).strip()

                    # Clean up response
                    if "Response:" in response:
                        response = response.split("Response:")[-1].strip()

                    logger.info(f"✅ Web Search: Generated response ({len(response)} chars)")
                    return response
                else:
                    logger.warning("Web Search: Empty response from LLM")
                    return "Unable to generate response from web search results. Please try again."

    except ImportError:
        logger.error("Web Search: duckduckgo_search module not available")
        return "Web search is not available. Please ensure duckduckgo-search is installed."
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")
        return f"Error performing web search: {str(e)}"

    finally:
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
