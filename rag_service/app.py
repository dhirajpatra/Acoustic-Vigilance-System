# rag_service/app.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import librosa # Still needed for audio loading/splitting
import numpy as np
# import faiss # Removed
# import torch # Removed (no Wav2Vec2/GPU tensor ops)
# import soundfile as sf # Can be useful, but librosa often includes backend
from tqdm import tqdm
import os
import io
import httpx # Still needed for health check / direct Ollama calls if needed
import logging
from dotenv import load_dotenv
from pydantic import BaseModel # Still needed for metadata validation

# LangChain Imports for Ollama, Chroma, etc.
from langchain_community.document_loaders import PyPDFLoader # Not needed for audio, but leaving as example from uploader.py
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter # Still useful if text descriptions get very long
from langchain_community.embeddings import OllamaEmbeddings # Use community version
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.documents import Document # For creating LangChain Documents

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Configuration ---
app = FastAPI(
    title="Acoustic Vigilance RAG Service (Ollama+Chroma Text Embeddings)",
    description="API for processing sound files by embedding textual descriptions of chunks using Ollama and storing in ChromaDB"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SoundMetadata(BaseModel):
    machine_type: str = "unknown machine"
    rpm: Optional[int] = None
    location: Optional[str] = "unknown location"
    notes: Optional[str] = "No additional notes"

class UploadResponse(BaseModel):
    status: str
    filename: str
    duration_seconds: float # Declare as float
    chunks_processed: int # Declare as int
    database: str
    embedding_source: str
    collection_count: int # Declare as int

# --- Constants ---
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
CHUNK_SIZE_SECONDS = int(os.getenv("CHUNK_SIZE_SECONDS", 30))
MAX_DURATION_MINUTES = int(os.getenv("MAX_DURATION_MINUTES", 3))
# EMBEDDING_DIM and OLLAMA_EMBEDDING_DIM are less critical as Chroma infers dimension

# --- Storage Paths (Similar to uploader.py & previous app.py) ---
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", "./storage/rag_audio"))
PERSIST_DIR = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")) # Using CHROMA_DB_PATH from env
EMBEDDING_CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_PATH", "./embedding_cache")) # New env var for cache

# Ensure storage paths exist
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
logging.info(f"Storage path ensured: {STORAGE_PATH}")
logging.info(f"Chroma persistence path ensured: {PERSIST_DIR}")
logging.info(f"Embedding cache path ensured: {EMBEDDING_CACHE_DIR}")


# --- Initialize Ollama Embeddings and ChromaDB with LangChain ---
try:
    # Ollama Embedding Model Configuration
    # Ensure these env vars are set in your .env
    ollama_embedding_model_name = os.getenv("EMBEDDING_MODEL") # Use EMBEDDING_MODEL from env
    ollama_server_url = os.getenv("OLLAMA_HOST") # Use OLLAMA_HOST from env

    if not ollama_embedding_model_name or not ollama_server_url:
        raise ValueError("EMBEDDING_MODEL and OLLAMA_HOST environment variables must be set for Ollama embeddings.")

    logging.info(f"Initializing OllamaEmbeddings with model '{ollama_embedding_model_name}' and server '{ollama_server_url}'")
    ollama_embeddings = OllamaEmbeddings(
        model=ollama_embedding_model_name,
        base_url=ollama_server_url,
        show_progress=True # Optional: show progress bar for embedding calls
    )

    # Setup Embedding Cache
    fs = LocalFileStore(EMBEDDING_CACHE_DIR)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        ollama_embeddings,
        fs,
        namespace=ollama_embedding_model_name # Namespace cache by model name
    )
    logging.info("CacheBackedEmbeddings initialized.")


    # ChromaDB Vectorstore Initialization (LangChain wrapper)
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "sound-text-chunks") # Use CHROMA_COLLECTION_NAME from env
    logging.info(f"Initializing Chroma vectorstore with persistence '{PERSIST_DIR}' and collection '{collection_name}'")

    # Initialize or load the persistent Chroma vectorstore
    # Note: Chroma.from_documents is for initial creation/adding.
    # For adding to an existing one, you get the collection/client first.
    # LangChain's Chroma wrapper can often handle persistence automatically via persist_directory.
    # We initialize the wrapper pointing to the directory.
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR), # Pass as string
        collection_name=collection_name,
        embedding_function=cached_embeddings # Use the cached Ollama embeddings
    )
    logging.info(f"Chroma vectorstore '{collection_name}' initialized/loaded. Approx count: {vectorstore._collection.count()}") # Accessing internal count for logging


except Exception as e:
    logging.error(f"Service Initialization failed: {str(e)}", exc_info=True)
    # Re-raise the exception after logging
    raise RuntimeError(f"Service Initialization failed: {str(e)}")


# --- Helper Functions ---
def validate_audio_duration(file_path: Path) -> float:
    """Validate audio file duration using librosa."""
    try:
        # Use 'path' argument as recommended in newer librosa versions
        duration = librosa.get_duration(path=str(file_path))
        if duration > MAX_DURATION_MINUTES * 60:
            raise ValueError(f"File exceeds {MAX_DURATION_MINUTES} minute limit ({duration:.2f}s)")
        return duration
    except Exception as e:
        logging.error(f"Audio duration validation failed for {file_path}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio file or duration exceeds limit: {str(e)}"
        )

# We no longer need process_audio_chunk or get_ollama_embedding as separate steps for ADDING

# --- API Endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload_sound_description(
    file: UploadFile = File(...),
    metadata: Optional[SoundMetadata] = None # Metadata about the sound file
):
    """
    Upload a sound file, split it into chunks, create text descriptions for each chunk
    based on metadata, embed descriptions using Ollama, and store in ChromaDB.
    """
    logging.info(f"Received upload request for file: {file.filename}, initial metadata: {metadata}")

    # Process metadata - use provided or default
    processed_metadata = metadata if metadata else SoundMetadata()
    logging.info(f"Using processed metadata: {processed_metadata.dict()}")

    try:
        # --- 1. Save the uploaded audio file ---
        file_path = STORAGE_PATH / file.filename
        # Ensure parent directory exists (already done on startup, but safe check)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Read file content asynchronously and save
        try:
            file_content = await file.read()
            with open(file_path, "wb") as f:
                f.write(file_content)
            logging.info(f"File saved to {file_path}")
        except Exception as e:
             logging.error(f"Failed to save file {file.filename}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

        # --- 2. Load audio and validate duration ---
        try:
            audio, sr = librosa.load(str(file_path), sr=SAMPLE_RATE)
            duration = validate_audio_duration(file_path) # Validate duration
            logging.info(f"Audio loaded from {file_path}, duration: {duration:.2f}s, sample rate: {sr}")
        except Exception as e:
             # librosa load or validate failed - clean up the saved file
             if file_path.exists():
                 file_path.unlink()
                 logging.info(f"Cleaned up saved file: {file_path}")
             # Re-raise or raise new HTTPException
             if isinstance(e, HTTPException): raise e # Re-raise validation errors
             logging.error(f"Failed to load or validate audio {file.filename}: {e}", exc_info=True)
             raise HTTPException(status_code=400, detail=f"Failed to load or process audio file: {e}")


        # --- 3. Split audio into chunks and create text descriptions ---
        chunk_size_samples = int(CHUNK_SIZE_SECONDS * sr)
        chunks_data = [] # To store info needed for creating Documents
        logging.info(f"Splitting audio into chunks of approx {CHUNK_SIZE_SECONDS} seconds ({chunk_size_samples} samples)...")

        for i in range(0, len(audio), chunk_size_samples):
            start_time = i / sr
            end_time = min((i + chunk_size_samples) / sr, duration)

            # Create a text description for the chunk
            # This description will be embedded by Ollama
            description = f"Audio chunk {i//chunk_size_samples + 1} from file '{file.filename}'."
            description += f" Time range: {start_time:.2f}s to {end_time:.2f}s."

            # Include metadata in the description text
            description += f" Machine Type: {processed_metadata.machine_type}."
            if processed_metadata.rpm is not None:
                description += f" RPM: {processed_metadata.rpm}."
            if processed_metadata.location:
                 description += f" Location: {processed_metadata.location}."
            if processed_metadata.notes and processed_metadata.notes != "No additional notes": # Avoid adding default notes unless they were user provided
                description += f" Notes: {processed_metadata.notes}."

            # Prepare metadata dictionary for ChromaDB document
            chunk_metadata_dict = {
                "original_file": file.filename,
                "chunk_index": i // chunk_size_samples,
                "start_time_seconds": start_time,
                "end_time_seconds": end_time,
                # Include all original metadata fields from the Pydantic model
                **processed_metadata.dict() # Flatten Pydantic model to dict
            }
            # Ensure no None values in metadata dict for Chroma (Chroma sometimes has issues with None)
            chunk_metadata_dict = {k: v for k, v in chunk_metadata_dict.items() if v is not None}


            # Create a LangChain Document for this chunk's description and metadata
            chunk_document = Document(page_content=description, metadata=chunk_metadata_dict)
            chunks_data.append(chunk_document)

        logging.info(f"Created {len(chunks_data)} text descriptions/documents for chunks.")


        # --- 4. Add documents (with embedded descriptions) to ChromaDB ---
        if not chunks_data:
            logging.warning(f"No documents generated for file {file.filename}. Skipping addition to DB.")
            raise HTTPException(status_code=400, detail="Could not generate processable chunks/descriptions for the audio file.")

        logging.info(f"Adding {len(chunks_data)} documents to ChromaDB collection '{vectorstore._collection.name}'...") # Access internal name

        try:
            # vectorstore.add_documents handles embedding the page_content using
            # the initialized embedding_function (cached_embeddings -> ollama_embeddings)
            # and adds the documents (page_content + metadata) to the collection.
            vectorstore.add_documents(chunks_data)
            logging.info(f"Successfully added {len(chunks_data)} documents to ChromaDB.")
            logging.info(f"Total documents in collection: {vectorstore._collection.count()}") # Access internal count

        except Exception as e:
            logging.error(f"Failed to add documents to ChromaDB for {file.filename}: {e}", exc_info=True)
            # Decide whether to clean up the saved audio file on DB error
            # if file_path.exists():
            #     file_path.unlink()
            #     logging.info(f"Cleaned up saved file {file_path} after DB error.")
            raise HTTPException(status_code=500, detail=f"Failed to store embeddings in database: {e}")


        # --- 5. Return success response ---
        return {
            "status": "success",
            "filename": file.filename,
            "duration_seconds": duration,
            "chunks_processed": len(chunks_data),
            "database": "ChromaDB",
            "embedding_source": "Ollama (Text Description)",
             "collection_count": vectorstore._collection.count() # Include updated count
        }

    except HTTPException:
        # Re-raise intentional HTTP exceptions (e.g., validation errors, DB errors)
        raise
    except Exception as e:
        # Catch any other unexpected errors during processing
        logging.error(f"Unexpected error during /upload processing for {file.filename}: {str(e)}", exc_info=True)
        # Clean up saved file on unexpected error
        if 'file_path' in locals() and file_path.exists():
            try:
                file_path.unlink()
                logging.info(f"Cleaned up saved file {file_path} after unexpected error.")
            except Exception as cleanup_e:
                logging.error(f"Failed to cleanup file {file_path}: {cleanup_e}")

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """System health check: Checks Ollama connection and ChromaDB accessibility."""
    components_status = {
        "ollama": False,
        "chromadb": False,
        "storage_path_writable": False,
        "persist_path_writable": False,
        "cache_path_writable": False,
    }
    error_detail = []
    status = "degraded"

    # Check paths exist and are writable
    components_status["storage_path_writable"] = STORAGE_PATH.exists() and os.access(STORAGE_PATH, os.W_OK)
    components_status["persist_path_writable"] = PERSIST_DIR.exists() and os.access(PERSIST_DIR, os.W_OK)
    components_status["cache_path_writable"] = EMBEDDING_CACHE_DIR.exists() and os.access(EMBEDDING_CACHE_DIR, os.W_OK)
    if not components_status["storage_path_writable"]: error_detail.append(f"Storage path not writable: {STORAGE_PATH}")
    if not components_status["persist_path_writable"]: error_detail.append(f"Chroma persist path not writable: {PERSIST_DIR}")
    if not components_status["cache_path_writable"]: error_detail.append(f"Embedding cache path not writable: {EMBEDDING_CACHE_DIR}")


    # Check Ollama connection
    ollama_server_url = os.getenv("OLLAMA_HOST")
    if ollama_server_url:
        async with httpx.AsyncClient() as client:
            try:
                # Use a simple endpoint like /api/tags or just the base URL
                ollama_resp = await client.get(f"{ollama_server_url}/api/tags", timeout=5.0)
                components_status["ollama"] = ollama_resp.status_code == 200
                if not components_status["ollama"]:
                     error_detail.append(f"Ollama check failed: Status code {ollama_resp.status_code}")
                else:
                    logging.info("Health check: Ollama is reachable.")

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                error_detail.append(f"Ollama connection failed: {e}")
    else:
         error_detail.append("OLLAMA_HOST environment variable not set.")


    # Check ChromaDB accessibility (try getting collection count)
    try:
        # Access the underlying collection via the vectorstore wrapper
        # This implicitly tests the client and collection
        chroma_count = vectorstore._collection.count()
        components_status["chromadb"] = True
        logging.info(f"Health check: ChromaDB collection '{vectorstore._collection.name}' accessible. Count: {chroma_count}")
    except Exception as e:
        error_detail.append(f"ChromaDB access failed: {e}")
        components_status["chromadb"] = False


    # Determine overall status
    required_components_healthy = (
        components_status["ollama"] and
        components_status["chromadb"] and
        components_status["storage_path_writable"] and
        components_status["persist_path_writable"] # Cache path is less critical for core function
    )

    if required_components_healthy:
        status = "healthy"
    elif any(components_status.values()):
        status = "partially_healthy"
    else:
        status = "unhealthy"


    response_content = {
        "status": status,
        "components": components_status
    }
    if error_detail:
         response_content["errors"] = error_detail

    # Return 500 if not fully healthy
    status_code = 200 if status == "healthy" else 500

    return JSONResponse(content=response_content, status_code=status_code)


# Add a root endpoint
@app.get("/")
async def read_root():
    return {"message": "Acoustic Vigilance RAG Service is running. Use /docs for API documentation."}