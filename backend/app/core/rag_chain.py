"""
RAG chain implementation using LangChain and Ollama
"""

import time
from typing import List, Dict, Any, Optional
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from backend.app.core.config import settings
from backend.app.core.vectorizer import vector_store
from backend.app.vector.dbs.milvus import MilvusClient
from backend.app.utils.logger import logger
import requests


class RAGChain:
    """RAG (Retrieval-Augmented Generation) chain"""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.prompt_template = None
        self.chain = None
        self.milvus_client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize LLM and prompt template"""
        try:
            self.milvus_client = MilvusClient()
            # Initialize Ollama LLM
            self.llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.ollama_temperature,
                streaming=True
            )

            # Initialize embeddings model
            self.embeddings = OllamaEmbeddings(
                base_url=settings.ollama_base_url,
                model=settings.ollama_embedding_model,
            )
            
            # Define prompt template
            self.prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""基于以下上下文信息，请回答用户的问题。如果上下文中没有相关信息，请诚实地说明无法从提供的文档中找到答案。

上下文信息：
{context}

用户问题：{question}

请提供详细且准确的回答："""
            )
            
            # Create LLM chain
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template,
                verbose=False
            )
            
            logger.info("RAG chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {str(e)}")
            raise e
    
    def query(self, question: str, collection_name: str = "milvus_test_collection", top_k: int = None, filter_metadata: Dict[str, Any] = None, model: str = None) -> Dict[str, Any]:
        """Process RAG query"""
        start_time = time.time()
        self.llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=model,
                temperature=settings.ollama_temperature,
                streaming=True
            )
        try:
            # Step 1: Retrieve relevant documents
            if top_k is None:
                top_k = settings.retrieval_top_k
            
            if settings.vector_db == 'milvus':
                query_vectrors = self.embeddings.embed_query(question)
                search_results = self.milvus_client.search(
                    collection_name=collection_name,
                    vectors=[query_vectrors],
                    limit=top_k
                )
                retrieved_docs = []
                counts = len(search_results.ids) or len(search_results.documents) or len(search_results.metadatas) or len(search_results.distances)
                for i in range(counts):
                    if not search_results.documents or len(search_results.documents) <= i:
                        continue
                    num_items = len(search_results.documents[i])
                    for j in range(num_items):
                        try:
                            metadata = search_results.metadatas[i][j]
                            content = search_results.documents[i][j]
                            similarity_score = search_results.distances[i][j]/1000
                            rank = j + 1
                            retrieved_docs.append({
                                'content': content,
                                'metadata': metadata,
                                'similarity_score': similarity_score,
                                'rank': rank
                            })
                        except Exception as e:
                            logger.error(f"Error processing document {i}: {str(e)}")
                            continue
                logger.info(f"Retrieved {len(retrieved_docs)} results for query: {question[:50]}...")
            else:
                retrieved_docs = vector_store.search(
                    query=question,
                    top_k=top_k,
                    filter_metadata=filter_metadata
                )
            
            if not retrieved_docs:
                return {
                    'question': question,
                    'answer': '抱歉，我无法在文档库中找到与您问题相关的信息。请尝试重新表述您的问题或上传相关文档。',
                    'sources': [],
                    'processing_time': time.time() - start_time,
                    'retrieved_chunks': 0
                }
            
            # Step 2: Prepare context from retrieved documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(retrieved_docs):
                # Add document content to context
                context_parts.append(f"文档片段 {i+1}:\n{doc['content']}\n")
                
                # Prepare source information
                source_info = {
                    'rank': doc['rank'],
                    'filename': doc['metadata'].get('filename', 'unknown'),
                    'file_type': doc['metadata'].get('file_type', 'unknown'),
                    'similarity_score': doc['similarity_score'],
                    'content_preview': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content']
                }
                sources.append(source_info)
            
            context = "\n".join(context_parts)
            
            # Step 3: Generate answer using LLM
            yield {'type': 'sources', 'data': sources}
            answer_parts = []
            formatted_prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            for chunk in self.llm.stream(formatted_prompt):
                text = chunk.content
                if text:
                    answer_parts.append(text)
                    yield {"type": "chunk", "data": text}
            # full_answer = "".join(answer_parts).strip()
            
            processing_time = time.time() - start_time
            
            logger.info(f"RAG query processed in {processing_time:.2f}s: {question[:50]}...")
            yield {
            'type': 'complete',
            'data': {
                'retrieved_chunks': len(retrieved_docs),
                'processing_time': time.time() - start_time
            }
        }
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {str(e)}")
            yield {
                'question': question,
                'answer': f'处理您的问题时发生错误：{str(e)}',
                'sources': [],
                'processing_time': time.time() - start_time,
                'retrieved_chunks': 0,
                'error': str(e)
            }
    
    def query_with_conversation_history(self, question: str, history: List[Dict[str, str]] = None, 
                                     top_k: int = None, filter_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process RAG query with conversation history"""
        start_time = time.time()
        
        try:
            # Enhance question with conversation context if available
            enhanced_question = question
            if history:
                # Add recent conversation context
                context_parts = []
                for turn in history[-3:]:  # Use last 3 turns for context
                    if 'question' in turn and 'answer' in turn:
                        context_parts.append(f"Q: {turn['question']}\nA: {turn['answer']}")
                
                if context_parts:
                    conversation_context = "\n\n".join(context_parts)
                    enhanced_question = f"基于以下对话历史：\n{conversation_context}\n\n当前问题：{question}"
            
            # Use regular query method with enhanced question
            result = self.query(enhanced_question, top_k, filter_metadata)
            
            # Update processing time
            result['processing_time'] = time.time() - start_time
            result['original_question'] = question
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing conversational RAG query: {str(e)}")
            return {
                'question': question,
                'answer': f'处理您的问题时发生错误：{str(e)}',
                'sources': [],
                'processing_time': time.time() - start_time,
                'retrieved_chunks': 0,
                'error': str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check RAG chain health"""
        try:
            # Test LLM connection
            response = requests.get(f"http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                test_response = response.json()
            else:
                test_response = None
            # test_response = self.llm.invoke("Hello")
            
            # Test vector store
            vector_health = vector_store.health_check()
            
            return {
                'healthy': True,
                'llm_model': settings.ollama_model,
                'llm_responsive': bool(test_response),
                'vector_store_healthy': vector_health.get('healthy', False),
                'embedding_model': settings.ollama_embedding_model
            }
            
        except Exception as e:
            logger.error(f"RAG chain health check failed: {str(e)}")
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def get_prompt_template(self) -> str:
        """Get current prompt template"""
        return self.prompt_template.template
    
    def update_prompt_template(self, new_template: str) -> bool:
        """Update prompt template"""
        try:
            self.prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=new_template
            )
            
            # Recreate chain with new template
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template,
                verbose=False
            )
            
            logger.info("Prompt template updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating prompt template: {str(e)}")
            return False
        
    def get_model_list(self) -> list:
        try:
            response = requests.get(f"http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return models
            else:
                logger.error("❌ 获取模型失败:", response.text)
                return []
        except Exception as e:
            logger.error("❌ 请求错误:", str(e))
            return []

# Global RAG chain instance
rag_chain = RAGChain()
