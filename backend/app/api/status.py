"""
System status API endpoints
"""

import time
import psutil
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import SystemStatus
from backend.app.core.vectorizer import vector_store
from backend.app.vector.dbs.milvus import milvus_client
from backend.app.core.rag_chain import rag_chain
from backend.app.utils.file_utils import get_disk_usage
from backend.app.core.config import settings
from backend.app.utils.logger import logger

router = APIRouter()

# Store startup time for uptime calculation
startup_time = time.time()

@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Check Ollama availability
        ollama_health = rag_chain.health_check()
        ollama_available = ollama_health.get('healthy', False)
        
        # Check Milvus availability
        vector_health = milvus_client.health_check()
        milvus_available = vector_health.get('healthy', False)
        
        # Get collection statistics
        stats = milvus_client.get_collection_stats()
        logger.info(f'collection info: {stats.get("collections_info", [])}')
        # Get disk usage
        disk_usage = get_disk_usage(settings.chroma_persist_directory)
        
        # Calculate uptime
        uptime_seconds = time.time() - startup_time
        uptime_str = str(timedelta(seconds=int(uptime_seconds)))
        
        # Overall system status
        overall_status = "healthy" if (ollama_available and milvus_available) else "degraded"
        
        return SystemStatus(
            status=overall_status,
            ollama_available=ollama_available,
            milvus_available=milvus_available,
            total_documents=stats.get('total_docs', 0),
            disk_usage=disk_usage,
            uptime=uptime_str,
            collections_info=stats.get("collections_info", [])
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@router.get("/status/health")
async def health_check():
    """Simple health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "RAG System"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/detailed")
async def get_detailed_status():
    """Get detailed system status including performance metrics"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Ollama status
        ollama_health = rag_chain.health_check()
        
        # Vector store status
        vector_health = vector_store.health_check()
        vector_stats = vector_store.get_collection_stats()
        
        # Disk usage for data directories
        data_disk_usage = get_disk_usage(settings.chroma_persist_directory)
        upload_disk_usage = get_disk_usage(settings.upload_dir)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": str(timedelta(seconds=int(time.time() - startup_time))),
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 2)
                }
            },
            "ollama": {
                "status": "healthy" if ollama_health.get('healthy', False) else "unhealthy",
                "model": settings.ollama_model,
                "embedding_model": settings.ollama_embedding_model,
                "base_url": settings.ollama_base_url,
                "details": ollama_health
            },
            "vector_store": {
                "status": "healthy" if vector_health.get('healthy', False) else "unhealthy",
                "total_documents": vector_stats.get('unique_documents', 0),
                "total_chunks": vector_stats.get('total_chunks', 0),
                "file_types": vector_stats.get('file_types', {}),
                "collection_name": settings.chroma_collection_name,
                "details": vector_health
            },
            "storage": {
                "vector_db": data_disk_usage,
                "uploads": upload_disk_usage
            },
            "configuration": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "max_file_size_mb": settings.max_file_size_mb,
                "supported_extensions": settings.allowed_extensions,
                "retrieval_top_k": settings.retrieval_top_k
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting detailed status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get detailed status: {str(e)}")

@router.get("/status/components")
async def get_component_status():
    """Get individual component status"""
    try:
        components = {}
        
        # Test each component individually
        try:
            ollama_health = rag_chain.health_check()
            components['ollama'] = {
                'status': 'healthy' if ollama_health.get('healthy', False) else 'unhealthy',
                'details': ollama_health
            }
        except Exception as e:
            components['ollama'] = {
                'status': 'error',
                'error': str(e)
            }
        
        try:
            vector_health = vector_store.health_check()
            components['chromadb'] = {
                'status': 'healthy' if vector_health.get('healthy', False) else 'unhealthy',
                'details': vector_health
            }
        except Exception as e:
            components['chromadb'] = {
                'status': 'error',
                'error': str(e)
            }
        
        try:
            from backend.app.core.document_parser import document_parser
            components['document_parser'] = {
                'status': 'healthy',
                'supported_extensions': document_parser.supported_extensions
            }
        except Exception as e:
            components['document_parser'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "components": components
        }
        
    except Exception as e:
        logger.error(f"Error getting component status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
