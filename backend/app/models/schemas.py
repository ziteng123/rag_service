"""
Pydantic models for API schemas
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Document metadata model"""
    filename: str
    file_type: str
    file_size: int
    upload_time: datetime
    chunk_count: Optional[int] = None
    author: Optional[str] = None
    title: Optional[str] = None

class DocumentChunk(BaseModel):
    """Document chunk model"""
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    embedding: Optional[List[float]] = None

class UploadResponse(BaseModel):
    """File upload response model"""
    status: str
    message: str
    file_count: int
    processed_files: List[str]
    failed_files: List[str] = []
    total_chunks: int = 0

class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., min_length=1, description="User question")
    collection_name: Optional[str] = Field(None, description="Collection name")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of top results to retrieve")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    model: Optional[str] = Field('qwen3:4b', description="LLM model to use for query processing")

class QueryResponse(BaseModel):
    """Query response model"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    retrieved_chunks: int

class SystemStatus(BaseModel):
    """System status model"""
    status: str
    ollama_available: bool
    milvus_available: bool
    total_documents: int
    disk_usage: Dict[str, Any]
    uptime: str
    collections_info: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: datetime

class DeleteRequest(BaseModel):
    """Delete request model"""
    document_ids: Optional[List[str]] = None
    delete_all: bool = False

class DeleteResponse(BaseModel):
    """Delete response model"""
    status: str
    message: str
    deleted_count: int
