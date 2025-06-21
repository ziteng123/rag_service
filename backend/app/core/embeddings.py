from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import hashlib
import uuid

from backend.app.core.config import settings
from backend.app.utils.logger import logger
from backend.app.vector.dbs.milvus import MilvusClient



class Embeddings:
    def __init__(self):
        self.client = None
        self.embedding = None
        self.text_splitter = None
        self._initialize()

    def _initialize(self):
        try:
            # Initialize Milvus client
            self.client = MilvusClient()
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            # Initialize embedding model
            self.embedding = OllamaEmbeddings(
                base_url=settings.ollama_base_url,
                model=settings.ollama_embedding_model
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise e

    def embed_to_milvus(self, item: dict) -> Dict[str, Any]:
        try:
            if not item['content'].strip():
                return {
                        'success': False,
                        'error': 'Empty document content',
                        'chunks_added': 0
                    }
            # Split document into chunks
            chunks = self.text_splitter.split_text(item['content'])
            if not chunks:
                    return {
                        'success': False,
                        'error': 'No chunks generated from document',
                        'chunks_added': 0
                    }
            filename = item['metadata'].filename
            
            total = len(chunks)
            for i in range(0, total, 1000):
                batch_chunks = chunks[i:i + 1000]
                docs = []
                # Generate embeddings for documents
                chunk_embeddings = self.embedding.embed_documents(batch_chunks)
                chunk_ids = []
                for idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    chunk_id = self._generate_chunk_id(filename, idx + i*1000)
                    chunk_ids.append(chunk_id)

                    chunk_metadata = {
                        'filename': filename,
                        'file_type': item['metadata'].file_type,
                        'file_size': item['metadata'].file_size,
                        'upload_time': item['metadata'].upload_time.isoformat(),
                        'chunk_index': idx + i*1000,
                        'chunk_count': len(chunks),
                        'content_preview': chunk[:100] + '...' if len(chunk) > 100 else chunk
                    }
                    if item['metadata'].author:
                        chunk_metadata['author'] = item['metadata'].author
                    if item['metadata'].title:
                        chunk_metadata['title'] = item['metadata'].title
                    doc = {
                        "id": chunk_id,
                        "text": chunk,
                        "vector": embedding,
                        "metadata": chunk_metadata
                    }
                    docs.append(doc)
                # Save to Milvus
                collection_name = "rag_service"+self._generate_chunk_id(filename, 0)
                self.client.upsert(collection_name, docs)
            logger.info(f"Added {len(chunks)} chunks for document: {filename}")
            return {
                    'success': True,
                    'chunks_added': len(chunks),
                    'chunk_ids': chunk_ids
                }
        except Exception as e:
            print(e)
            logger.error(f"Error adding document to vector store: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'chunks_added': 0
            }



    def _generate_chunk_id(self, filename: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        content = f"{filename}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    

    
embeddings = Embeddings()
