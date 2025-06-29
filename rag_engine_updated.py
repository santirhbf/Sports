import pandas as pd
from sentence_transformers import SentenceTransformer, util
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from datetime import datetime

import google.generativeai as genai
import streamlit as st

from config import config
from document_parser import DocumentProcessor
from country_classifier import CountryClassifier
from translator import TranslationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiResponse:
    def __init__(self, response: str, processing_time: float, 
                 context_used: str, error: Optional[str] = None):
        self.response = response
        self.processing_time = processing_time
        self.context_used = context_used
        self.error = error
        self.timestamp = datetime.now()

class RAGEngine:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.data_processor = PerformanceDataProcessor()
        self.country_classifier = CountryClassifier()
        self.translator = TranslationService()
        
        # Initialize Gemini client
        self.gemini_client = None
        self._initialize_gemini_client()
    
    def _initialize_gemini_client(self):
        """Initialize Gemini API client"""
        try:
            if config.GOOGLE_API_KEY and not config.GOOGLE_API_KEY.startswith("your-"):
                genai.configure(api_key=config.GOOGLE_API_KEY)
                self.gemini_client = genai.GenerativeModel(config.GEMINI_MODEL)
                logger.info("Gemini 2.0 Flash client initialized")
            else:
                logger.error("Gemini API key not configured")
                
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
    
    def _create_system_prompt(self, context: str) -> str:
        return f"""You are an elite sports performance assistant for physical trainers working with professional football players.

        The trainer has access to detailed player performance and recovery data, described below.

        CONTEXT FROM PERFORMANCE DATA:
        {context}

        INSTRUCTIONS:
        1. Answer the trainer's question using only the provided context
        2. Highlight practical, actionable recommendations where relevant
        3. If data is missing or inconclusive, say so
        4. Be concise but clear. Use metrics or thresholds mentioned in the context
        5. Use sport science language appropriate for high-performance environments
        6. If the trainer asks for recommendations, use your expertise + data context to respond

        Format your answer clearly and use bullet points when useful.
        """
    
    def _query_gemini(self, system_prompt: str, user_query: str) -> GeminiResponse:
        """Query Gemini 2.0 Flash model"""
        start_time = time.time()
        
        try:
            if not self.gemini_client:
                return GeminiResponse(
                    "", 0, "", 
                    "Gemini client not initialized - check API key"
                )
            
            # Combine system prompt and user query for Gemini
            full_prompt = f"{system_prompt}\n\nUSER QUESTION: {user_query}"
            
            response = self.gemini_client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE
                )
            )
            
            processing_time = time.time() - start_time
            answer = response.text
            
            return GeminiResponse(answer, processing_time, system_prompt)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Gemini query failed: {e}")
            return GeminiResponse("", processing_time, system_prompt, str(e))
    
    def _ensure_vector_stores_ready(self):
        """Ensure all vector stores are initialized"""
        if not self.doc_processor.vector_stores:
            logger.info("Initializing vector stores...")
            self.doc_processor.initialize_all_vector_stores()
    
    def query_player_data(self, query: str) -> Dict[str, Any]:
        """
        Query the player's performance and recovery data using Gemini.
        Returns the context used, model response, and timing.
     """
        original_query = query
        start_time = time.time()

        try:
            # Step 1: Get relevant rows from the merged dataset
            context, row_indices = self.data_processor.get_relevant_context(query)

            if not context:
                return {
                    "error": "No relevant data found",
                    "user_message": "No performance or recovery data matched the query. Try rephrasing.",
                    "original_query": original_query
                }

            # Step 2: Create the system prompt with the context
            system_prompt = self._create_system_prompt(context)

            # Step 3: Query Gemini with full prompt
            gemini_response = self._query_gemini(system_prompt, query)

            if gemini_response.error:
                return {
                    "error": f"Gemini query failed: {gemini_response.error}",
                    "original_query": original_query
                }

            # Step 4: Return everything
            return {
                "original_query": original_query,
                "context_rows": row_indices,
                "context_used": context,
                "gemini_response": {
                    "text": gemini_response.response,
                    "processing_time": gemini_response.processing_time
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "original_query": original_query
            }

class PerformanceDataProcessor:
    def __init__(self):
        self.df = pd.read_csv("merged_df.csv")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode(self.df.apply(lambda row: ' '.join(map(str, row.values)), axis=1), convert_to_tensor=True)
    
    def get_relevant_context(self, query: str, top_k: int = 5) -> Tuple[str, List[int]]:
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]
        relevant_rows = [self.df.iloc[hit['corpus_id']].to_dict() for hit in hits]
        context = "\n".join([json.dumps(row, indent=2) for row in relevant_rows])
        return context, [hit['corpus_id'] for hit in hits]

# rag_engine.py
# Utility functions for Streamlit integration
@st.cache_resource
def load_rag_engine():
    """Cached RAG engine for Streamlit"""
    return RAGEngine()

def format_gemini_response(response: Dict[str, Any]) -> str:
    """Format Gemini response for display"""
    if response.get("error"):
        return f"**Gemini 2.0 Flash** ❌\n*Error: {response['error']}*\n"
    
    processing_time = response.get("processing_time", 0)
    translated_response = response.get("translated_response", "")
    original_response = response.get("original_response", "")
    
    output = f"**Gemini 2.0 Flash** ✅\n"
    output += f"*Processing time: {processing_time:.2f}s*\n\n"
    
    if translated_response != original_response:
        output += f"**Translated Response:**\n{translated_response}\n\n"
        output += f"**Original Response:**\n{original_response}\n"
    else:
        output += f"{translated_response}\n"
    
    return output + "\n---\n"