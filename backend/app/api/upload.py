"""
File upload API endpoints
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from backend.app.models.schemas import UploadResponse, DeleteRequest, DeleteResponse
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

@router.get("/upload/documents")
async def get_uploaded_documents():
    """Get list of uploaded documents"""
    try:
        # Get all documents from vector store
        results = vector_store.collection.get(
            include=['metadatas']
        )
        
        # Group by filename to get unique documents
        documents = {}
        for metadata in results['metadatas']:
            filename = metadata.get('filename', 'unknown')
            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'file_type': metadata.get('file_type', 'unknown'),
                    'file_size': metadata.get('file_size', 0),
                    'upload_time': metadata.get('upload_time', ''),
                    'chunk_count': 0,
                    'author': metadata.get('author', ''),
                    'title': metadata.get('title', '')
                }
            documents[filename]['chunk_count'] += 1
        
        # Convert to list and sort by upload time
        document_list = list(documents.values())
        document_list.sort(key=lambda x: x['upload_time'], reverse=True)
        
        return {
            "status": "success",
            "documents": document_list,
            "total_count": len(document_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting uploaded documents: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@router.delete("/upload/documents/{filename}")
async def delete_document(filename: str):
    """Delete a specific document by filename"""
    try:
        # Delete document from vector store
        result = vector_store.delete_document(filename)
        
        if result['success']:
            return DeleteResponse(
                status="success",
                message=f"Successfully deleted document: {filename}",
                deleted_count=result['deleted_count']
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=result.get('error', f'Document not found: {filename}')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/upload/documents/delete", response_model=DeleteResponse)
async def delete_documents(request: DeleteRequest):
    """Delete multiple documents or all documents"""
    try:
        if request.delete_all:
            # Delete all documents
            try:
                vector_store.collection.delete()
                # Recreate collection
                vector_store.collection = vector_store.chroma_client.create_collection(
                    name=vector_store.collection.name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                return DeleteResponse(
                    status="success",
                    message="Successfully deleted all documents",
                    deleted_count=-1  # Indicate all documents deleted
                )
            except Exception as e:
                logger.error(f"Error deleting all documents: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to delete all documents: {str(e)}")
        
        elif request.document_ids:
            # Delete specific documents
            total_deleted = 0
            failed_deletions = []
            
            for filename in request.document_ids:
                result = vector_store.delete_document(filename)
                if result['success']:
                    total_deleted += result['deleted_count']
                else:
                    failed_deletions.append(filename)
            
            if failed_deletions:
                message = f"Deleted {total_deleted} chunks from {len(request.document_ids) - len(failed_deletions)} documents. Failed: {', '.join(failed_deletions)}"
            else:
                message = f"Successfully deleted {total_deleted} chunks from {len(request.document_ids)} documents"
            
            return DeleteResponse(
                status="success" if not failed_deletions else "partial",
                message=message,
                deleted_count=total_deleted
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Either document_ids must be provided or delete_all must be true"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk delete operation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
