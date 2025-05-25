"""
Vectorization module using Ollama embeddings and ChromaDB
"""

import hashlib
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

from backend.app.core.config import settings
from backend.app.utils.logger import logger
from backend.app.models.schemas import DocumentMetadata, DocumentChunk

class VectorStore:
    """Vector store using ChromaDB and Ollama embeddings"""
    
    def __init__(self):
        self.embeddings = None
        self.chroma_client = None
        self.collection = None
        self.text_splitter = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB and embedding model"""
        try:
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize Ollama embeddings
            self.embeddings = OllamaEmbeddings(
                base_url=settings.ollama_base_url,
                model=settings.ollama_embedding_model
            )
            
            # Initialize ChromaDB client
            persist_dir = Path(settings.chroma_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=settings.chroma_collection_name
                )
                logger.info(f"Loaded existing collection: {settings.chroma_collection_name}")
            except ValueError:
                self.collection = self.chroma_client.create_collection(
                    name=settings.chroma_collection_name,
                    metadata={"hnsw:space": settings.chroma_distance_function}
                )
                logger.info(f"Created new collection: {settings.chroma_collection_name}")
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise e
    
    def add_document(self, content: str, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Add document to vector store"""
        try:
            if not content.strip():
                return {
                    'success': False,
                    'error': 'Empty document content',
                    'chunks_added': 0
                }
            
            # Split document into chunks
            chunks = self.text_splitter.split_text(content)
            
            if not chunks:
                return {
                    'success': False,
                    'error': 'No chunks generated from document',
                    'chunks_added': 0
                }
            
            # Generate embeddings for chunks
            chunk_embeddings = self.embeddings.embed_documents(chunks)
            
            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                # Generate unique chunk ID
                chunk_id = self._generate_chunk_id(metadata.filename, i)
                chunk_ids.append(chunk_id)
                
                # Prepare metadata
                chunk_metadata = {
                    'filename': metadata.filename,
                    'file_type': metadata.file_type,
                    'file_size': metadata.file_size,
                    'upload_time': metadata.upload_time.isoformat(),
                    'chunk_index': i,
                    'chunk_count': len(chunks),
                    'content_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
                }
                
                if metadata.author:
                    chunk_metadata['author'] = metadata.author
                if metadata.title:
                    chunk_metadata['title'] = metadata.title
                
                chunk_metadatas.append(chunk_metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=chunk_ids,
                embeddings=chunk_embeddings,
                documents=chunks,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks for document: {metadata.filename}")
            
            return {
                'success': True,
                'chunks_added': len(chunks),
                'chunk_ids': chunk_ids
            }
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'chunks_added': 0
            }
    
    def search(self, query: str, top_k: int = None, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if top_k is None:
                top_k = settings.retrieval_top_k
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = {}
                for key, value in filter_metadata.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = {"$eq": value}
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Delete document chunks by filename"""
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"filename": {"$eq": filename}},
                include=['metadatas']
            )
            
            if not results['ids']:
                return {
                    'success': False,
                    'error': f'No chunks found for document: {filename}',
                    'deleted_count': 0
                }
            
            # Delete chunks
            self.collection.delete(ids=results['ids'])
            
            deleted_count = len(results['ids'])
            logger.info(f"Deleted {deleted_count} chunks for document: {filename}")
            
            return {
                'success': True,
                'deleted_count': deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'deleted_count': 0
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(
                limit=min(100, count),
                include=['metadatas']
            )
            
            # Count unique documents
            unique_files = set()
            file_types = {}
            
            for metadata in sample_results['metadatas']:
                filename = metadata.get('filename', 'unknown')
                file_type = metadata.get('file_type', 'unknown')
                
                unique_files.add(filename)
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            return {
                'total_chunks': count,
                'unique_documents': len(unique_files),
                'file_types': file_types,
                'collection_name': settings.chroma_collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                'total_chunks': 0,
                'unique_documents': 0,
                'file_types': {},
                'error': str(e)
            }
    
    def _generate_chunk_id(self, filename: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        content = f"{filename}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def health_check(self) -> Dict[str, Any]:
        """Check vector store health"""
        try:
            # Test embedding generation
            test_embedding = self.embeddings.embed_query("test")
            
            # Test ChromaDB connection
            count = self.collection.count()
            
            return {
                'healthy': True,
                'embedding_model': settings.ollama_embedding_model,
                'collection_count': count,
                'embedding_dimension': len(test_embedding)
            }
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
            return {
                'healthy': False,
                'error': str(e)
            }

# Global vector store instance
vector_store = VectorStore()
