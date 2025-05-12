# comparison_service/app.py

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa # Still needed for audio loading/duration
import numpy as np
# import faiss # Removed
# import torch # Removed (no Wav2Vec2 or GPU tensor ops)
from pathlib import Path
from dotenv import load_dotenv
import io
import time # Needed for wait_for_ollama_model
from typing import Dict, List, Optional, Tuple # Import Tuple
from pydantic import BaseModel # Ensure BaseModel is imported
import os
import httpx # Still needed for Ollama API calls (embedding check and LLM)
import json # Import json for parsing LLM output
import logging # Import logging
# import asyncio # Removed asyncio as we're using synchronous method directly

# LangChain Imports for Ollama Embeddings, ChromaDB, etc.
from langchain_community.embeddings import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma # LangChain Chroma wrapper
from langchain_core.documents import Document # Useful for structuring data, even query results

# Import chromadb client for type hinting if needed
# import chromadb

logging.basicConfig(level=logging.INFO) # Configure basic logging
logger = logging.getLogger(__name__) # Use logger for this module

# Configuration
load_dotenv()

# Correct FastAPI app declaration for comparison service
app = FastAPI(
    title="Acoustic Vigilance Comparison Service (Ollama+Chroma Text Embeddings)",
    description="API for comparing test sound files using Ollama text embeddings against ChromaDB and getting LLM analysis"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variables & Constants ---
# Ollama Embedding Model Configuration (MUST MATCH RAG SERVICE's EMBEDDING_MODEL)
# This model is used to EMBED THE TEXT DESCRIPTION of the test audio segment for querying Chroma.
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL") # Use EMBEDDING_MODEL from env
# Ollama LLM Model Configuration (Used for Analysis)
# This model is used for the final text generation analysis.
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL", "tinyllama:1.1b") # Use OLLAMA_MODEL from env for LLM
OLLAMA_HOST_URL = os.getenv("OLLAMA_HOST", "http://ollama_server:11434") # Use OLLAMA_HOST

# ChromaDB Configuration (MUST MATCH RAG SERVICE's CONFIG)
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "sound-text-chunks") # Use CHROMA_COLLECTION_NAME from env
EMBEDDING_CACHE_PATH = Path(os.getenv("EMBEDDING_CACHE_PATH", "./embedding_cache")) # Path for embedding cache

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7)) # Used for preliminary diagnosis

# Audio Processing Constants (Still relevant for describing the test audio chunk)
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
ANALYSIS_WINDOW_SEC = int(os.getenv("CHUNK_SIZE_SECONDS", 30)) # Use CHUNK_SIZE_SECONDS for the analysis window
# Text splitting constants from retriever.py template - not directly used for audio, but good to have if descriptions get long
# CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300)) # Not directly used for audio chunking
# CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50)) # Not directly used for audio chunking


# --- Global Initialization (Done Once at Startup) ---
# We initialize ChromaDB client/collection/wrapper and Ollama Embeddings here
ollama_embeddings_instance: Optional[OllamaEmbeddings] = None
cached_embeddings_instance: Optional[CacheBackedEmbeddings] = None
vectorstore: Optional[Chroma] = None # LangChain Chroma wrapper

# --- Helper Function from retriever.py template ---
def wait_for_ollama_model(model: str, base_url: str, timeout: int = 90, interval: int = 5):
    """Wait until the embedding model is available on Ollama."""
    # Check /api/tags endpoint as a general health check first
    tags_url = f"{base_url}/api/tags"
    embed_url = f"{base_url}/api/embed" # Specific endpoint for embeddings check
    dummy_payload = {"model": model, "prompt": "ping"}
    start_time = time.time()

    logger.info(f"Waiting for Ollama server at {base_url} and model '{model}' to be ready...")

    while time.time() - start_time < timeout:
        try:
            # First, check if the server is generally reachable
            tags_response = httpx.get(tags_url, timeout=5)
            tags_response.raise_for_status() # Raises for 4xx/5xx responses

            # If server is up, try the embed endpoint specifically with the model
            # Ollama might return 404 if the model is not pulled/available
            embed_response = httpx.post(embed_url, json=dummy_payload, timeout=5)
            if embed_response.status_code != 404:
                # Any status other than 404 suggests the model endpoint is being handled
                # 200 is ideal, but 400 might mean invalid prompt, but server/model are responsive.
                logger.info(f"Ollama server is up and model '{model}' endpoint is responsive.")
                return True # Model is ready

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            # Log specific HTTP or Request errors
            status_code_str = f" (Status: {e.response.status_code})" if isinstance(e, httpx.HTTPStatusError) else ""
            logger.warning(f"Ollama not ready yet or connectivity issue{status_code_str}: {e}")
        except Exception as e:
            # Catch other potential errors
            logger.warning(f"Unexpected error while waiting for Ollama: {e}")

        logger.info(f"Waiting for Ollama model '{model}' to load...")
        time.sleep(interval)

    # If loop finishes without returning, it timed out
    raise RuntimeError(f"Timeout waiting for Ollama server or model '{model}' to load.")

# --- Helper Function from retriever.py template (optional, but good for health check) ---
def is_chroma_db_initialized(persist_dir: Path) -> list[str]:
    """Check if ChromaDB persistence directory looks initialized."""
    # Check for the main sqlite file
    db_file = persist_dir / 'chroma.sqlite3'
    if not db_file.exists():
        # logger.info(f"ChromaDB file not found at {db_file}.") # Avoid spamming logs on health check
        return []

    # Check for collection directories (36 char UUIDs)
    # This is a heuristic, but indicative of a collection
    try:
        collection_dirs = [
            d.name for d in persist_dir.iterdir() if d.is_dir() and
            len(d.name) == 36 and all(c in '0123456789abcdef-' for c in d.name)
        ]
        # logger.info(f"Found {len(collection_dirs)} potential collection directories.")
        return collection_dirs
    except Exception as e:
        logger.warning(f"Error listing ChromaDB directories in {persist_dir}: {e}")
        return []


def load_services():
    """Loads necessary services (Chroma, Ollama Embeddings) into global variables."""
    global ollama_embeddings_instance, cached_embeddings_instance, vectorstore

    # --- Ensure paths exist ---
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Chroma persistence path ensured: {CHROMA_DB_PATH}")
    logger.info(f"Embedding cache path ensured: {EMBEDDING_CACHE_PATH}")

    # --- Wait for Ollama Embedding Model ---
    if not EMBEDDING_MODEL_ID or not OLLAMA_HOST_URL:
         logger.critical("FATAL: EMBEDDING_MODEL or OLLAMA_HOST environment variables must be set for Ollama embeddings.")
         raise RuntimeError("Ollama embedding configuration missing.")
    try:
        wait_for_ollama_model(EMBEDDING_MODEL_ID, OLLAMA_HOST_URL)
    except RuntimeError as e:
         logger.critical(f"FATAL: {e}")
         raise e # Re-raise timeout error


    # --- Ollama Embedding Initialization ---
    logger.info(f"Initializing OllamaEmbeddings with model '{EMBEDDING_MODEL_ID}' and server '{OLLAMA_HOST_URL}'")
    try:
        ollama_embeddings_instance = OllamaEmbeddings(
            model=EMBEDDING_MODEL_ID,
            base_url=OLLAMA_HOST_URL,
            show_progress=False # Avoid progress bars in logs
        )

        # Setup Embedding Cache
        fs = LocalFileStore(EMBEDDING_CACHE_PATH)
        cached_embeddings_instance = CacheBackedEmbeddings.from_bytes_store(
            ollama_embeddings_instance,
            fs,
            namespace=EMBEDDING_MODEL_ID # Namespace cache by model name
        )
        logger.info("CacheBackedEmbeddings initialized.")

        # --- ChromaDB Vectorstore Initialization (LangChain wrapper) ---
        # We initialize the wrapper pointing to the directory and collection name.
        # LangChain will handle loading the client and collection.
        logger.info(f"Initializing Chroma vectorstore with persistence '{CHROMA_DB_PATH}' and collection '{CHROMA_COLLECTION_NAME}'")
        vectorstore = Chroma(
             persist_directory=str(CHROMA_DB_PATH), # Pass as string
             collection_name=CHROMA_COLLECTION_NAME,
             embedding_function=cached_embeddings_instance # Use the cached Ollama embeddings
             # persist_directory is handled by passing the client
        )
        # Access the underlying collection via the protected member to check count/name during startup logging
        logger.info(f"LangChain Chroma vectorstore wrapper initialized/loaded for collection '{vectorstore._collection.name}'. Initial count: {vectorstore._collection.count()}")


    except Exception as e:
        logger.critical(f"FATAL: Ollama Embedding or Chroma Initialization failed: {str(e)}", exc_info=True)
        # If this fails, the service cannot query embeddings. It is fatal.
        raise RuntimeError(f"Service Initialization failed: {str(e)}")


# Load services when the script starts
try:
    load_services()
except RuntimeError:
    # If load_services fails (e.g., Ollama timeout, Chroma error), the app cannot function.
    # Endpoints will check if services are loaded using ensure_services_loaded.
    logger.error("Failed to load necessary services. Service will not be fully functional.")


# --- Pydantic Models for Responses ---

class LLMAnalysis(BaseModel):
    """Model for the LLM's structured analysis output."""
    technical_analysis: str
    recommended_actions: List[str]
    confidence: float
    error: Optional[str] = None # Optional field for LLM errors
    raw_llm_output: Optional[str] = None # Optional field for raw output on error


class ComparisonResult(BaseModel):
    """Model for the full comparison and LLM analysis result."""
    similarity: float # Calculated similarity score (0-1)
    preliminary_diagnosis: str  # Diagnosis based on similarity threshold
    llm_analysis: LLMAnalysis   # Structured analysis from the LLM
    processed_seconds: float # Duration of the audio segment processed
    top_matches_info: List[Dict] # Info about top matches from ChromaDB


class SimpleComparisonResult(BaseModel):
    """Model for the basic comparison result (without LLM)."""
    similarity: float
    diagnosis: str # Based on similarity threshold
    analysis_window: int # Duration of the analysis window in seconds
    processed_samples: int # Number of samples processed for the window
    top_matches_info: List[Dict] # Info about top matches from ChromaDB


# --- Core Processing Functions ---

def ensure_services_loaded():
    """Checks if core services (Chroma vectorstore, Cached Embeddings) are loaded, raises HTTPException if not."""
    if vectorstore is None or cached_embeddings_instance is None:
        logger.error("Core services (ChromaDB vectorstore, Cached Embeddings) not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Core services are not loaded. Service is not ready."
        )

async def generate_text_embedding_for_test_audio(audio_description: str) -> List[float]:
     """Generates an embedding for a text description using the cached Ollama embeddings."""
     ensure_services_loaded() # Ensure cached_embeddings_instance is available
     try:
         # LangChain's embed_query handles caching and calling the underlying model
         # It returns List[float]
         # NOTE: This is an async method (aembed_query)
         embedding = await cached_embeddings_instance.aembed_query(audio_description)
         logger.info(f"Generated embedding for description using '{EMBEDDING_MODEL_ID}'. Embedding dimension: {len(embedding)}")
         return embedding
     except Exception as e:
         logger.error(f"Failed to generate embedding for audio description: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate embedding: {str(e)}")


# Return type hint matches the output of similarity_search_with_score_by_vector
async def search_chromadb(query_embedding: List[float], k: int = 3) -> List[Tuple[Document, float]]:
    """
    Queries ChromaDB with a generated embedding using the lower-level query method.
    Returns a list of (Document, distance) tuples.
    """
    ensure_services_loaded()

    chroma_count = vectorstore._collection.count()
    if chroma_count == 0:
        logger.warning("ChromaDB collection is empty. Cannot perform search.")
        return []

    logger.info(f"Performing ChromaDB search with k={k} against {chroma_count} documents.")
    try:
        raw_results = vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )

        documents = raw_results.get("documents", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]

        results: List[Tuple[Document, float]] = [
            (Document(page_content=doc, metadata=meta), dist)
            for doc, dist, meta in zip(documents, distances, metadatas)
        ]

        logger.info(f"ChromaDB search successful. Found {len(results)} results.")
        return results

    except Exception as e:
        logger.error(f"ChromaDB search failed using collection.query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ChromaDB search failed: {str(e)}"
        )


async def get_llm_diagnosis(similarity: float, top_matches_info: List[Dict], processed_seconds: float) -> LLMAnalysis:
    """
    Gets a detailed analysis from the LLM using Ollama API directly via httpx.
    Returns an LLMAnalysis object, handling potential errors internally.
    """
    # Prepare information about top matches for the prompt (limit to first few)
    # top_matches_info contains metadata, distance, and page_content
    top_matches_for_prompt = top_matches_info[:3] # Limit information sent to LLM

    prompt = f"""
    Analyze machine sound based on an acoustic similarity comparison score and reference matches from a database.
    The test audio segment was {processed_seconds:.2f} seconds long.
    - Calculated Similarity Score: {similarity:.2f}/1.0 (1.0 = perfect match).
    - Configured Threshold: {SIMILARITY_THRESHOLD}. Scores > threshold typically suggest normal operation or a known state. Scores <= threshold may indicate a potential anomaly.
    - Closest Database Matches: {json.dumps(top_matches_for_prompt)} (Each match represents a chunk of reference audio. It includes its original text description, metadata like original file, chunk index, machine type, notes, time range, etc., and a 'distance' score. Lower distance means higher similarity).

    Perform a technical analysis based on the similarity score and the content (text description and metadata) of the closest matches.
    - Provide a likely diagnosis (e.g., "Normal Operation", "Potential Anomaly - Bearing Wear", "Unknown Anomaly"). Reference details from the closest matches if relevant.
    - Recommend maintenance or investigation actions based on the diagnosis.
    - Assess your confidence in the diagnosis (0.0 to 1.0).

    Structure your entire response STRICTLY as a JSON object with these keys:
    {{
        "technical_analysis": "Your concise analysis and diagnosis summary (50-150 words).",
        "recommended_actions": ["Action 1", "Action 2", ...],
        "confidence": 0.0 to 1.0
    }}
    """

    # Ensure Ollama LLM configuration is available
    if not OLLAMA_HOST_URL or not OLLAMA_MODEL_ID:
        logger.error("Ollama LLM host or model ID not configured.")
        return LLMAnalysis(
            technical_analysis="Ollama LLM configuration missing. Cannot perform detailed analysis.",
            recommended_actions=[], confidence=0.0, error="Ollama LLM not configured."
        )

    logger.info(f"Sending analysis prompt to Ollama LLM model '{OLLAMA_MODEL_ID}' at '{OLLAMA_HOST_URL}'...")
    # logger.debug(f"Prompt:\n{prompt}") # Log the full prompt for debugging if needed

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_HOST_URL}/api/generate", # Use /api/generate for text generation
                json={
                    "model": OLLAMA_MODEL_ID,
                    "prompt": prompt,
                    "format": "json", # Request JSON format output from Ollama
                    "options": {
                        "temperature": 0.3, # Keep temperature relatively low
                        "num_predict": -1, # Generate until the model stops
                         "top_p": 0.9 # Example option
                    }
                },
                timeout=90.0 # Increased timeout for LLM response
            )
            response.raise_for_status() # Raise an exception for bad status codes

            # Ollama API /api/generate returns JSON with a "response" key containing the generated text
            response_data = response.json()
            llm_output_text = response_data.get("response", "").strip()
            logger.info("Received response from Ollama LLM. Attempting to parse JSON.")
            # logger.debug(f"Raw LLM output:\n{llm_output_text}") # Log raw output for debugging

            # Attempt to parse the 'response' text as JSON
            try:
                llm_json_output = json.loads(llm_output_text)
                # Validate required keys are present and types are plausible
                analysis_text = llm_json_output.get("technical_analysis")
                actions_list = llm_json_output.get("recommended_actions")
                confidence_score = llm_json_output.get("confidence")

                if not isinstance(analysis_text, str) or not isinstance(actions_list, list) or not isinstance(confidence_score, (int, float)):
                     missing_or_invalid = []
                     if not isinstance(analysis_text, str): missing_or_invalid.append("technical_analysis (expected string)")
                     if not isinstance(actions_list, list): missing_or_invalid.append("recommended_actions (expected list)")
                     if not isinstance(confidence_score, (int, float)): missing_or_invalid.append("confidence (expected number)")

                     raise ValueError(f"LLM response JSON missing required keys or has invalid types: {', '.join(missing_or_invalid)}. Raw: {llm_json_output}")

                # Return the parsed and validated LLMAnalysis object
                return LLMAnalysis(
                    technical_analysis=analysis_text,
                    recommended_actions=actions_list,
                    confidence=float(confidence_score) # Ensure confidence is float
                )

            except (json.JSONDecodeError, ValueError) as e:
                 # Handle cases where LLM output is not valid JSON or missing/invalid keys
                 error_msg = f"LLM output parsing/validation failed: {e}"
                 logger.error(error_msg, exc_info=True)
                 return LLMAnalysis(
                    technical_analysis=f"LLM response format error: {e}. Raw output received.",
                    recommended_actions=["Review LLM prompt and model's JSON formatting ability."],
                    confidence=0.0,
                    error=error_msg,
                    raw_llm_output=llm_output_text
                 )

    except httpx.HTTPStatusError as e:
         # Handle HTTP errors from Ollama API
         error_msg = f"Ollama API HTTP error ({e.response.status_code}): {e.response.text}"
         logger.error(error_msg, exc_info=True)
         return LLMAnalysis(
            technical_analysis=error_msg,
            recommended_actions=["Check Ollama service status, logs, and model availability."],
            confidence=0.0,
            error=f"Ollama API HTTP error: {e.response.status_code}"
         )
    except httpx.RequestError as e:
        # Handle network or request errors when calling Ollama
        error_msg = f"Ollama API request error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return LLMAnalysis(
            technical_analysis=error_msg,
            recommended_actions=["Check Ollama service status and network connectivity."],
            confidence=0.0,
            error=f"Ollama API request error: {str(e)}"
        )
    except Exception as e:
        # Handle any other unexpected errors during the LLM call process
        error_msg = f"An unexpected error occurred during LLM analysis call: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return LLMAnalysis(
            technical_analysis=error_msg,
            recommended_actions=["Review service logs for LLM call errors."],
            confidence=0.0,
            error=f"Unexpected error: {str(e)}"
        )

# --- API Endpoints ---

@app.post("/compare", response_model=SimpleComparisonResult)
async def compare(file: UploadFile = File(...)) -> SimpleComparisonResult:
    """Basic comparison against ChromaDB using Ollama text embeddings."""
    logger.info(f"Received basic comparison request for file: {file.filename}")
    try:
        ensure_services_loaded() # Ensure services are loaded

        audio_bytes = await file.read()
        try:
            audio, sr = librosa.load(
                io.BytesIO(audio_bytes),
                sr=SAMPLE_RATE,
                duration=ANALYSIS_WINDOW_SEC
            )
        except Exception as e:
            logger.error(f"Failed to load audio for comparison: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load or process audio file: {str(e)}"
            )

        target_samples = ANALYSIS_WINDOW_SEC * sr
        processed_audio_segment = audio
        if len(processed_audio_segment) < target_samples:
             processed_audio_segment = np.pad(processed_audio_segment, (0, max(0, target_samples - len(processed_audio_segment))), mode='constant')
        elif len(processed_audio_segment) > target_samples:
             processed_audio_segment = processed_audio_segment[:target_samples]
        processed_seconds = len(processed_audio_segment) / sr

        audio_description = f"An audio recording of a machine sound segment, duration {processed_seconds:.2f} seconds."

        query_embedding = await generate_text_embedding_for_test_audio(audio_description)

        # search_chromadb returns List[Tuple[Document, float]]
        # NOTE: This is now a synchronous search method called directly
        results = await search_chromadb(query_embedding, k=3) # Call search_chromadb

        # --- Process ChromaDB search results from List[Tuple[Document, float]] format ---
        # mean_similarity is assigned inside the if/else block
        if not results: # Check if the list is empty
            mean_similarity = 0.0 # Assigned here if no results
            top_matches_info = [] # Assigned here if no results
            logger.warning("ChromaDB search returned no results.")
        else:
            # results is List[Tuple[Document, float]] -> [(doc1, dist1), (doc2, dist2), ...]
            distances = [score for doc, score in results]
            # Calculate mean similarity
            mean_distance = np.mean(distances)
            mean_similarity = 1 / (1 + mean_distance) # Assuming L2-like distance, assigned here if results exist

            # Prepare top matches info
            top_matches_info = []
            for doc, distance in results:
                 match_info = {
                     "id": doc.metadata.get('id', 'N/A'), # Get ID from metadata or generate if missing
                     "distance": float(distance),
                     "metadata": doc.metadata, # Include all metadata
                     "page_content": doc.page_content # Include the original text description of the reference chunk
                 }
                 top_matches_info.append(match_info)
        # --- End Process ---


        diagnosis_status = "healthy" if mean_similarity > SIMILARITY_THRESHOLD else "faulty"

        logger.info(f"Comparison successful. Similarity: {mean_similarity:.2f}, Diagnosis: {diagnosis_status}")

        return SimpleComparisonResult(
            similarity=float(mean_similarity),
            diagnosis=diagnosis_status,
            analysis_window=ANALYSIS_WINDOW_SEC,
            processed_samples=len(processed_audio_segment),
            top_matches_info=top_matches_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"Unexpected error during /compare processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during comparison: {str(e)}"
        )


@app.post("/analyze", response_model=ComparisonResult)
async def analyze_audio(file: UploadFile = File(...)):
    """Comprehensive analysis with ChromaDB comparison and LLM diagnosis."""
    logger.info(f"Received comprehensive analysis request for file: {file.filename}")
    try:
        ensure_services_loaded() # Ensure services are loaded

        audio_bytes = await file.read()
        try:
             audio, sr = librosa.load(
                 io.BytesIO(audio_bytes),
                 sr=SAMPLE_RATE,
                 duration=ANALYSIS_WINDOW_SEC
             )
        except Exception as e:
             logger.error(f"Failed to load audio for analysis: {e}", exc_info=True)
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 detail=f"Failed to load or process audio file: {str(e)}"
             )

        target_samples = ANALYSIS_WINDOW_SEC * sr
        processed_audio_segment = audio
        if len(processed_audio_segment) < target_samples:
             processed_audio_segment = np.pad(processed_audio_segment, (0, max(0, target_samples - len(processed_audio_segment))), mode='constant')
        elif len(processed_audio_segment) > target_samples:
             processed_audio_segment = processed_audio_segment[:target_samples]
        processed_seconds = len(processed_audio_segment) / sr

        audio_description = f"An audio recording of a machine sound segment, duration {processed_seconds:.2f} seconds."

        query_embedding = await generate_text_embedding_for_test_audio(audio_description)

        # search_chromadb returns List[Tuple[Document, float]]
        # NOTE: This is now a synchronous search method called directly
        results = await search_chromadb(query_embedding, k=3) # Call search_chromadb

        # --- Process ChromaDB search results from List[Tuple[Document, float]] format ---
        # Initialize variables before conditional
        mean_similarity = 0.0
        top_matches_info = []

        if not results: # Check if the list is empty
            logger.warning("ChromaDB search returned no results for analysis.")
            # If no results, return a default analysis immediately without calling LLM
            no_data_llm_analysis = LLMAnalysis(
                 technical_analysis="No reference data available for comparison.",
                 recommended_actions=["Ensure reference data is uploaded via the RAG service."],
                 confidence=0.0,
                 error="No reference data."
            )
            # Return the ComparisonResult with default/empty info and the specific LLM analysis error
            return ComparisonResult(
                 similarity=0.0, # Similarity is 0 if no matches
                 preliminary_diagnosis="no_reference_data",
                 llm_analysis=no_data_llm_analysis, # Pass the analysis indicating no data
                 processed_seconds=processed_seconds,
                 top_matches_info=[] # Explicitly empty list
            )

        else:
            # results is List[Tuple[Document, float]] -> [(doc1, dist1), (doc2, dist2), ...]
            distances = [score for doc, score in results]
            # Calculate mean similarity
            mean_distance = np.mean(distances)
            mean_similarity = 1 / (1 + mean_distance) # Assuming L2-like distance

            # Prepare top matches info
            for doc, distance in results:
                 match_info = {
                     "id": doc.metadata.get('id', 'N/A'), # Get ID from metadata or generate if missing
                     "distance": float(distance),
                     "metadata": doc.metadata, # Include all metadata
                     "page_content": doc.page_content # Include the original text description of the reference chunk
                 }
                 top_matches_info.append(match_info)
        # --- End Process ---

        # --- preliminary_diagnosis_status calculation now uses the potentially updated mean_similarity ---
        preliminary_diagnosis_status = "healthy" if mean_similarity > SIMILARITY_THRESHOLD else "faulty"
        # --- END preliminary_diagnosis_status ---


        logger.info(f"Comparison successful. Similarity: {mean_similarity:.2f}, Preliminary Diagnosis: {preliminary_diagnosis_status}. Calling LLM for detailed analysis.")

        # 6. Call LLM for detailed analysis
        llm_analysis_result = await get_llm_diagnosis(
            mean_similarity, # Pass the mean similarity
            top_matches_info=top_matches_info, # Pass top matches info including metadata and page_content
            processed_seconds=processed_seconds
        )

        # 7. Return the combined result
        return ComparisonResult(
            similarity=float(mean_similarity),
            preliminary_diagnosis=preliminary_diagnosis_status,
            llm_analysis=llm_analysis_result,
            processed_seconds=processed_seconds,
            top_matches_info=top_matches_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"Unexpected error during /analyze processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during analysis: {str(e)}"
        )


@app.get("/config")
async def get_config():
    """Return current service configuration and status of loaded components."""
    # Check if core services are loaded
    services_loaded = vectorstore is not None and cached_embeddings_instance is not None

    # Access attributes safely only if services are loaded
    chroma_collection_name = vectorstore._collection.name if vectorstore and vectorstore._collection else "Not loaded"
    chroma_count = vectorstore._collection.count() if vectorstore and vectorstore._collection else "Not loaded"
    embedding_model_id = EMBEDDING_MODEL_ID if EMBEDDING_MODEL_ID else "Not configured"
    ollama_llm_model_id = OLLAMA_MODEL_ID if OLLAMA_MODEL_ID else "Not configured"
    ollama_host_url_cfg = OLLAMA_HOST_URL if OLLAMA_HOST_URL else "Not configured"
    embedding_dim = "N/A (Chroma infers)" # Chroma infers dimension

    # Attempt to get embedding dimension if possible (e.g., by embedding a dummy query)
    # This is more complex for CacheBackedEmbeddings, might skip or do during init log
    # For now, rely on Chroma inferring it.

    return {
        "sample_rate": SAMPLE_RATE,
        "analysis_window_sec": ANALYSIS_WINDOW_SEC,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "embedding_model_id_for_comparison_query": embedding_model_id, # Report which embedding model is used for queries
        "ollama_llm_model_id_for_analysis": ollama_llm_model_id, # Report which LLM model is used for analysis
        "ollama_host_url": ollama_host_url_cfg,
        "chroma_db_path": str(CHROMA_DB_PATH),
        "embedding_cache_path": str(EMBEDDING_CACHE_PATH),
        "loaded_components": {
            "core_services_loaded": services_loaded, # General check for Chroma/Embeddings
        },
        "chroma_info": {
            "collection_name": chroma_collection_name,
            "total_documents": chroma_count,
        },
        "embedding_info": {
            "model_id": embedding_model_id,
            "dimension": embedding_dim, # Report N/A or try to get it
            "cache_path": str(EMBEDDING_CACHE_PATH),
        },
        "gpu_available": False # Assuming no direct torch/cuda usage now
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check: Checks core services (Chroma, Ollama Embeddings) and Ollama LLM connection."""
    health_status_detail: Dict = {}
    errors = []
    overall_healthy = True

    # 1. Check Core Services Loading Status (Chroma vectorstore, Cached Embeddings)
    core_services_loaded = vectorstore is not None and cached_embeddings_instance is not None
    health_status_detail["core_services_loaded"] = core_services_loaded
    if not core_services_loaded:
        errors.append("Core services (ChromaDB vectorstore, Cached Embeddings) failed to load at startup.")
        overall_healthy = False
    else:
         health_status_detail["chroma_db_path"] = str(CHROMA_DB_PATH)
         health_status_detail["chroma_collection_name"] = CHROMA_COLLECTION_NAME
         health_status_detail["embedding_model_id"] = EMBEDDING_MODEL_ID
         health_status_detail["embedding_cache_path"] = str(EMBEDDING_CACHE_PATH)


    # If core services didn't load, report unhealthy immediately
    if not overall_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "details": health_status_detail, "errors": errors}
        )

    # 2. Check ChromaDB Accessibility (try getting collection count)
    try:
        # Access the underlying collection via the vectorstore wrapper
        # This implicitly tests the client and collection
        chroma_count = vectorstore._collection.count() # Use the underlying collection object
        components_status["chromadb"] = True
        health_status_detail["chroma_document_count"] = chroma_count
        logger.info(f"Health check: ChromaDB collection '{vectorstore._collection.name}' accessible. Count: {chroma_count}")
    except Exception as e:
        errors.append(f"ChromaDB access health check failed: {e}")
        components_status["chromadb"] = False
        overall_healthy = False # Mark unhealthy if Chroma access fails

    # 3. Check Ollama Embedding Model Connection (using the wait_for_ollama_model logic)
    ollama_embed_connection_healthy = False
    if EMBEDDING_MODEL_ID and OLLAMA_HOST_URL:
        health_status_detail["ollama_embedding_model_id"] = EMBEDDING_MODEL_ID
        health_status_detail["ollama_host_url"] = OLLAMA_HOST_URL
        try:
            # Use the wait logic with a short timeout just for health check
            wait_for_ollama_model(EMBEDDING_MODEL_ID, OLLAMA_HOST_URL, timeout=5, interval=1)
            ollama_embed_connection_healthy = True
            health_status_detail["ollama_embedding_connection"] = True

        except Exception as e:
            errors.append(f"Ollama embedding model health check failed ({EMBEDDING_MODEL_ID} at {OLLAMA_HOST_URL}): {str(e)}")
            health_status_detail["ollama_embedding_connection"] = False
            health_status_detail["ollama_embedding_error"] = str(e)
            overall_healthy = False # Mark unhealthy if embedding model is not reachable

    else:
        health_status_detail["ollama_embedding_connection"] = False
        errors.append("EMBEDDING_MODEL or OLLAMA_HOST environment variable not set for embeddings.")
        overall_healthy = False


    # 4. Check Ollama LLM Connection (using httpx directly as before)
    ollama_llm_connection_healthy = False
    if OLLAMA_HOST_URL and OLLAMA_MODEL_ID:
        health_status_detail["ollama_llm_model_id"] = OLLAMA_MODEL_ID
        # OLLAMA_HOST_URL is already reported above
        try:
            async with httpx.AsyncClient() as client:
                # Check /api/tags or base URL
                ollama_resp = await client.get(f"{OLLAMA_HOST_URL}/api/tags", timeout=5.0) # Use short timeout for health check
                ollama_resp.raise_for_status() # Raise for bad status

                ollama_llm_connection_healthy = True
                health_status_detail["ollama_llm_connection"] = True
                health_status_detail["ollama_llm_status"] = ollama_resp.status_code

        except Exception as e:
            errors.append(f"Ollama LLM connection health check failed ({OLLAMA_MODEL_ID} at {OLLAMA_HOST_URL}): {str(e)}")
            health_status_detail["ollama_llm_connection"] = False
            health_status_detail["ollama_llm_error"] = str(e)
            overall_healthy = False # Mark unhealthy if Ollama connection fails
    else:
        health_status_detail["ollama_llm_connection"] = False
        errors.append("OLLAMA_HOST_URL or OLLAMA_MODEL_ID environment variable not set for LLM.")
        overall_healthy = False


    # Determine overall status and return response
    # Service is healthy only if all components check out
    if overall_healthy:
        status_code = status.HTTP_200_OK
        final_status = "healthy"
    else:
         status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
         final_status = "unhealthy"


    response_content = {"status": final_status, "details": health_status_detail}
    if errors:
         response_content["errors"] = errors

    return JSONResponse(status_code=status_code, content=response_content)


# Add a root endpoint
@app.get("/")
async def read_root():
    return {"message": "Acoustic Vigilance Comparison Service is running. Use /docs for API documentation."}