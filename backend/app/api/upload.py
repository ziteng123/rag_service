"""
File upload API endpoints
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from backend.app.models.schemas import UploadResponse
from backend.app.core.document_parser import document_parser
from backend.app.core.vectorizer import vector_store
from backend.app.utils.file_utils import save_multiple_files, cleanup_temp_files, validate_file_type, validate_file_size
from backend.app.utils.logger import logger

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and process multiple files"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate files
        for file in files:
            if not validate_file_type(file.filename):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            # Read file size for validation
            content = await file.read()
            await file.seek(0)  # Reset file pointer
            
            if not validate_file_size(len(content)):
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {file.filename}"
                )
        
        # Save files to disk
        save_results = await save_multiple_files(files)
        
        if not save_results['saved_files']:
            raise HTTPException(
                status_code=500,
                detail="Failed to save any files"
            )
        
        # Process saved files
        processed_files = []
        failed_files = []
        total_chunks = 0
        file_paths_to_cleanup = []
        
        for file_info in save_results['saved_files']:
            file_path = file_info['file_path']
            filename = file_info['filename']
            file_paths_to_cleanup.append(file_path)
            
            try:
                # Parse document
                parse_result = document_parser.parse_document(file_path)
                
                if not parse_result['success']:
                    failed_files.append(filename)
                    logger.error(f"Failed to parse {filename}: {parse_result.get('error', 'Unknown error')}")
                    continue
                
                # Add to vector store
                vector_result = vector_store.add_document(
                    content=parse_result['content'],
                    metadata=parse_result['metadata']
                )
                
                if vector_result['success']:
                    processed_files.append(filename)
                    total_chunks += vector_result['chunks_added']
                    logger.info(f"Successfully processed {filename}: {vector_result['chunks_added']} chunks")
                else:
                    failed_files.append(filename)
                    logger.error(f"Failed to vectorize {filename}: {vector_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                failed_files.append(filename)
                logger.error(f"Error processing {filename}: {str(e)}")
        
        # Schedule cleanup of temporary files
        background_tasks.add_task(cleanup_temp_files, file_paths_to_cleanup)
        
        # Prepare response
        status = "success" if processed_files else "failed"
        message = f"Processed {len(processed_files)} files successfully"
        
        if failed_files:
            message += f", {len(failed_files)} files failed"
        
        return UploadResponse(
            status=status,
            message=message,
            file_count=len(processed_files),
            processed_files=processed_files,
            failed_files=failed_files,
            total_chunks=total_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/upload/status")
async def get_upload_status():
    """Get upload system status"""
    try:
        # Get vector store statistics
        stats = vector_store.get_collection_stats()
        
        return {
            "status": "healthy",
            "total_documents": stats.get('unique_documents', 0),
            "total_chunks": stats.get('total_chunks', 0),
            "file_types": stats.get('file_types', {}),
            "supported_formats": document_parser.supported_extensions
        }
        
    except Exception as e:
        logger.error(f"Error getting upload status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )
