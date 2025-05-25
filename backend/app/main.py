"""
FastAPI main application for RAG system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uvicorn

from backend.app.core.config import settings
from backend.app.utils.logger import logger
from backend.app.api import upload, query, status
from backend.app.models.schemas import ErrorResponse
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting RAG System API...")
    logger.info(f"Configuration loaded: Ollama={settings.ollama_base_url}, ChromaDB={settings.chroma_persist_directory}")
    
    try:
        from backend.app.core.vectorizer import vector_store
        from backend.app.core.rag_chain import rag_chain
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise e

    yield  # Application runs here

    # Shutdown logic
    logger.info("Shutting down RAG System API...")

# Create FastAPI application
app = FastAPI(
    lifespan=lifespan,
    title="RAG System API",
    description="Retrieval-Augmented Generation system with multi-format document support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "status": "/api/v1/status"
    }

@app.get("/api/v1")
async def api_info():
    """API information endpoint"""
    return {
        "name": "RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/v1/upload",
            "query": "/api/v1/query",
            "status": "/api/v1/status"
        },
        "documentation": "/docs"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
            timestamp=datetime.now()
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            timestamp=datetime.now()
        ).model_dump()
    )



if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
