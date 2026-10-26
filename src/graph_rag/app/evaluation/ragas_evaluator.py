from typing import Optional, Dict, List
import logging
import pandas as pd
from app.llm.llm import get_llm
from agents.agent import search_documents, get_agent_response
from config import config
import os
import json
from langchain.prompts import PromptTemplate
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

def load_ragas_template():
    try:
        df = pd.read_csv(config.RAGAS_DATA_PATH)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to load RAGAs template: {str(e)}")
        return []

def get_ground_truth(query: str, ragas_templates: List[Dict]) -> str:
    """Retrieve ground truth from RAGAs_Template.csv based on the query."""
    for template in ragas_templates:
        if template.get("question", "").lower().strip() == query.lower().strip():
            return template.get("ground_truths", "").strip()
    logger.warning(f"No ground truth found for query: {query}")
    return ""

def evaluate_ragas(query: str, answer: str, context: str, doc_id: Optional[str] = None) -> Dict:
    try:
        llm = get_llm()
        ragas_templates = load_ragas_template()

        # Search documents to get context and source papers
        docs, context, source_papers = search_documents(query)
        primary_paper_id = source_papers[0].get("id") if source_papers else None

        # Generate response if not provided
        response = answer if answer else get_agent_response(query, context, doc_id=primary_paper_id)
        logger.debug(f"Generated response: {response}")

        # Get ground truth
        ground_truth = get_ground_truth(query, ragas_templates)

        # Clean the response before evaluation
        cleaned_response = re.sub(r'\s*\(doc_id=[^)]+, chunk_id=[^)]+\)\s*', '', response).strip()
        logger.debug(f"Cleaned response for evaluation: {cleaned_response}")

        # Filter context to only include the chunk used in the answer (if available)
        answer_chunk_id = re.search(r'chunk_id=([^)]+)', response)
        filtered_context = context  # Default to full context
        is_kg_derived = False
        if answer_chunk_id:
            chunk_id = answer_chunk_id.group(1)
            filtered_context_lines = [doc.page_content for doc in docs if doc.metadata.get("chunk_id") == chunk_id]
            filtered_context = filtered_context_lines[0] if filtered_context_lines else context
            is_kg_derived = True  # Mark as KG-derived if chunk_id is present
        logger.debug(f"Filtered context: {filtered_context[:500]}...")

        # Detailed prompt for metric-based scoring with KG awareness
        judge_prompt = PromptTemplate.from_template(
            """
You are an evaluation expert assessing the quality of a response to a query based on provided context and ground truth. Score each metric from 0.0 to 1.0 (0.0 = poor, 1.0 = excellent) based on the following criteria. Return ONLY a JSON object with the scores, no extra text.

- context_precision: How well the provided context supports the answer. 1.0 if all context is relevant, 0.0 if none is relevant, partial scores for mixed relevance.
- context_recall: How much of the relevant information needed to answer the query is present in the context. 1.0 if all needed info is present, 0.0 if critical info is missing, partial scores for partial coverage.
- answer_relevancy: How relevant the answer is to the query. 1.0 if fully relevant, 0.0 if unrelated, partial scores if partially relevant.
- faithfulness: How well the answer aligns with the ground truth, especially if derived from context. 1.0 if identical or minor variation, 0.0 if contradictory, partial scores for partial alignment. If the answer is derived from context (KG), prioritize alignment with ground truth over context alone.

Query: {query}
Context: {context}
Answer: {answer}
Ground Truth: {ground_truth}
"""
        )

        # Invoke LLM to judge the response using cleaned response
        judge_response = llm.invoke(judge_prompt.format(query=query, context=filtered_context[:500], answer=cleaned_response, ground_truth=ground_truth))
        response_text = judge_response.content if hasattr(judge_response, 'content') else str(judge_response)
        logger.debug(f"Raw LLM response: {response_text}")

        # Handle malformed responses
        if response_text.strip() in ['"context_precision"', '"context_recall"', '"answer_relevancy"', '"faithfulness"']:
            logger.warning(f"LLM returned key name only: {response_text}")
            response_text = '{"context_precision": 0.0, "context_recall": 0.0, "answer_relevancy": 0.0, "faithfulness": 0.0}'

        match = re.search(r'\{[^}]*\}', response_text, re.DOTALL)
        if match:
            response_text = match.group(0)
        else:
            logger.warning(f"No valid JSON found, using fallback: {response_text[:200]}")
            response_text = '{"context_precision": 0.0, "context_recall": 0.0, "answer_relevancy": 0.0, "faithfulness": 0.0}'

        try:
            scores = json.loads(response_text)
            required_scores = {"context_precision": 0.0, "context_recall": 0.0, "answer_relevancy": 0.0, "faithfulness": 0.0}
            scores = {k: float(v) for k, v in {**required_scores, **scores}.items() if k in required_scores}
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {response_text[:200]}... Error: {str(e)}")
            scores = {"context_precision": 0.0, "context_recall": 0.0, "answer_relevancy": 0.0, "faithfulness": 0.0}

        # Post-process scores for consistency
        if "I cannot find enough information" in response:
            scores["answer_relevancy"] = 0.0  # Fallback response is not relevant to the query
            scores["faithfulness"] = 0.0  # Fallback is faithful to the system but not to the query—set to 0 for out-of-context
            scores["context_recall"] = 0.0 if ground_truth and ground_truth not in context else scores["context_recall"]
            scores["context_precision"] = 0.0 if not docs else scores["context_precision"]

        # Adjust faithfulness based on KG derivation and similarity
        if ground_truth and response != "I cannot find enough information":
            similarity = SequenceMatcher(None, cleaned_response, ground_truth).ratio()
            if is_kg_derived:
                # For KG-derived responses, set faithfulness based on similarity with a strict threshold
                scores["faithfulness"] = 1.0 if similarity > 0.95 else max(scores["faithfulness"], similarity * 0.8)
            scores["answer_relevancy"] = max(scores["answer_relevancy"], similarity)

        # For out-of-context, set doc_id to null if no relevant docs
        if not docs or "I cannot find enough information" in response:
            primary_paper_id = None

        evaluation_results = {
            "query": query,
            "answer": cleaned_response,
            "scores": scores,
            "timestamp": pd.Timestamp.now().isoformat(),
            "doc_id": primary_paper_id,
            "ground_truth": ground_truth
        }

        output_dir = os.path.dirname(config.RAGAS_OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)
        with open(config.RAGAS_OUTPUT_PATH, 'a') as f:
            json.dump(evaluation_results, f)
            f.write('\n')

        return evaluation_results
    except Exception as e:
        logger.error(f"RAGAs evaluation failed: {str(e)}")
        return {"error": str(e)}

def get_ragas_evaluation_history() -> List[Dict]:
    if os.path.exists(config.RAGAS_OUTPUT_PATH):
        with open(config.RAGAS_OUTPUT_PATH, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]
    return []
