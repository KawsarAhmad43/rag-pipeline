

####--------------------------------####
from typing import Optional

def text_generation_prompt(
    query: str,
    context: str,
    history: Optional[str] = "",
    for_speech: bool = False,
) -> str:
    """
    Creates a professional, ChatGPT-like prompt for document-based and conversational queries.
    """
    context = context.strip() if context else ""
    history = history.strip() if history else ""

    speak_guidance = (
        "\n\n[TTS GUIDANCE: Keep responses oral-friendly, avoid complex markdown, and use clear, short sentences.]"
        if for_speech else ""
    )

    prompt = f"""
You are a sophisticated AI assistant designed to provide exceptionally helpful and accurate information. Your goal is to mimic the high-quality conversational flow of advanced industrial models.

[CORE INSTRUCTIONS]
1. Context Priority: If "Document Context" is provided, use it as your primary source of truth.
2. Intelligent Retrieval: If the user asks for a "summary" or is asking about the document generally, analyze all provided context segments to synthesize a comprehensive overview.
3. General Knowledge Fallback: If the document context does not contain the answer, use your extensive general knowledge to assist the user. Be honest and mention that the document doesn't explicitly state the answer, but provide helpful information anyway.
4. Professionalism: Maintain a warm, professional, and direct tone. Use Markdown (lists, bolding) for clarity when appropriate.{speak_guidance}

[CONVERSATION HISTORY]
{history if history else "No previous history."}

[DOCUMENT CONTEXT]
---
{context if context else "No specific document context provided for this turn."}
---

[USER QUERY]
"{query}"

Assistant Response:
"""
    return prompt.strip()


def multimodal_prompt(prompt_text: str, for_speech: bool = False) -> str:
    """
    Guides the model through analyzing images/documents with a professional persona.
    """
    base_request = (prompt_text or "").strip()
    speak_guidance = (
        "\n[TTS GUIDANCE: Keep your analysis concise and use spoken-word rhythms.]"
        if for_speech else ""
    )

    return f"""
You are analyzing a visual document or image with high precision.

[ANALYSIS PROTOCOL]
1. Objective Review: Describe the visual content accurately based on the user's request.
2. Meta-Data Awareness: If asked about symbols, text within the image, or document structure, be specific.
3. Conversational Interface: Treat greetings and general questions naturally.
4. Synthesis: If professional document context is provided alongside the image, synthesize the two for a holistic answer.{speak_guidance}

[USER REQUEST]
"{base_request}"

Assistant Response:
"""


def error_prompt(error_message: str) -> str:
    """Generate a high-quality error fallback."""
    return f"I apologize, but I encountered a technical hurdle while processing that: {error_message}. Could you please try again or rephrase your request? I'm here to help."
