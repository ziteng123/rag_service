"""
Query API endpoints for RAG system
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from backend.app.models.schemas import QueryRequest, QueryResponse
from backend.app.core.rag_chain import rag_chain
from backend.app.utils.logger import logger

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using RAG"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Process RAG query
        result = rag_chain.query(
            question=request.question,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )
        
        return QueryResponse(
            question=result['question'],
            answer=result['answer'],
            sources=result['sources'],
            processing_time=result['processing_time'],
            retrieved_chunks=result['retrieved_chunks']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """Query documents with streaming response"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        async def generate_response():
            try:
                # First, send the retrieval status
                yield f"data: {json.dumps({'type': 'status', 'message': '正在检索相关文档...'})}\n\n"
                
                # Process RAG query
                result = rag_chain.query(
                    question=request.question,
                    top_k=request.top_k,
                    filter_metadata=request.filter_metadata,
                    model=request.model
                )
                for item in result:
                    if item.get('type') == 'sources':
                        yield f"data: {json.dumps({'type': 'sources', 'sources': item['data']})}\n\n"
                    if item.get('type') == 'chunk':
                        yield f"data: {json.dumps({'type': 'answer', 'answer': item['data']})}\n\n"
                    if item.get('type') == 'complete':
                        yield f"data: {json.dumps({'type': 'complete', 'processing_time': item['data']['processing_time'], 'retrieved_chunks': item['data']['retrieved_chunks']})}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming query: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up streaming query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/query/conversation")
async def query_with_history(
    question: str,
    history: Optional[list] = None,
    top_k: Optional[int] = None,
    filter_metadata: Optional[Dict[str, Any]] = None
):
    """Query with conversation history"""
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Process conversational RAG query
        result = rag_chain.query_with_conversation_history(
            question=question,
            history=history or [],
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        return {
            "question": result['question'],
            "answer": result['answer'],
            "sources": result['sources'],
            "processing_time": result['processing_time'],
            "retrieved_chunks": result['retrieved_chunks']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing conversational query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/query/health")
async def query_health_check():
    """Check query system health"""
    try:
        health_status = rag_chain.health_check()
        
        return {
            "status": "healthy" if health_status.get('healthy', False) else "unhealthy",
            "details": health_status
        }
        
    except Exception as e:
        logger.error(f"Query health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/query/prompt")
async def get_prompt_template():
    """Get current prompt template"""
    try:
        template = rag_chain.get_prompt_template()
        return {"prompt_template": template}
        
    except Exception as e:
        logger.error(f"Error getting prompt template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/prompt")
async def update_prompt_template(new_template: str):
    """Update prompt template"""
    try:
        if not new_template.strip():
            raise HTTPException(status_code=400, detail="Template cannot be empty")
        
        success = rag_chain.update_prompt_template(new_template)
        
        if success:
            return {"status": "success", "message": "Prompt template updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update prompt template")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt template: {str(e)}")

@router.get("/query/models")
async def query_models():
    """Query available models"""
    try:
        models = rag_chain.get_model_list()
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error querying models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
