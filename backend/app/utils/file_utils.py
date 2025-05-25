"""
File handling utilities
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
import aiofiles
from fastapi import UploadFile

from backend.app.core.config import settings
from backend.app.utils.logger import logger

async def save_uploaded_file(file: UploadFile, upload_dir: str = None) -> Dict[str, Any]:
    """Save uploaded file to disk"""
    try:
        if upload_dir is None:
            upload_dir = settings.upload_dir
        
        # Create upload directory if it doesn't exist
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        file_extension = Path(file.filename).suffix
        base_name = Path(file.filename).stem
        counter = 1
        final_filename = file.filename
        
        while (upload_path / final_filename).exists():
            final_filename = f"{base_name}_{counter}{file_extension}"
            counter += 1
        
        file_path = upload_path / final_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        return {
            'success': True,
            'file_path': str(file_path),
            'filename': final_filename,
            'original_filename': file.filename,
            'file_size': len(content)
        }
        
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'filename': file.filename
        }

async def save_multiple_files(files: List[UploadFile], upload_dir: str = None) -> Dict[str, Any]:
    """Save multiple uploaded files"""
    results = {
        'success': True,
        'saved_files': [],
        'failed_files': [],
        'total_files': len(files)
    }
    
    for file in files:
        result = await save_uploaded_file(file, upload_dir)
        
        if result['success']:
            results['saved_files'].append({
                'filename': result['filename'],
                'original_filename': result['original_filename'],
                'file_path': result['file_path'],
                'file_size': result['file_size']
            })
        else:
            results['failed_files'].append({
                'filename': result['filename'],
                'error': result['error']
            })
            results['success'] = False
    
    return results

def cleanup_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get file information"""
    try:
        path = Path(file_path)
        if not path.exists():
            return {'exists': False}
        
        stat = path.stat()
        return {
            'exists': True,
            'filename': path.name,
            'file_size': stat.st_size,
            'file_extension': path.suffix.lower(),
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime
        }
        
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        return {'exists': False, 'error': str(e)}

def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported"""
    file_extension = Path(filename).suffix.lower()
    return file_extension in settings.allowed_extensions

def validate_file_size(file_size: int) -> bool:
    """Validate if file size is within limits"""
    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    return file_size <= max_size_bytes

def get_disk_usage(directory: str) -> Dict[str, Any]:
    """Get disk usage information for directory"""
    try:
        path = Path(directory)
        if not path.exists():
            return {'exists': False}
        
        total, used, free = shutil.disk_usage(path)
        
        return {
            'exists': True,
            'total_bytes': total,
            'used_bytes': used,
            'free_bytes': free,
            'total_gb': round(total / (1024**3), 2),
            'used_gb': round(used / (1024**3), 2),
            'free_gb': round(free / (1024**3), 2),
            'usage_percent': round((used / total) * 100, 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting disk usage for {directory}: {str(e)}")
        return {'exists': False, 'error': str(e)}

def list_files_in_directory(directory: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
    """List files in directory with optional extension filtering"""
    try:
        path = Path(directory)
        if not path.exists():
            return []
        
        files = []
        for file_path in path.iterdir():
            if file_path.is_file():
                if extensions is None or file_path.suffix.lower() in extensions:
                    stat = file_path.stat()
                    files.append({
                        'filename': file_path.name,
                        'file_path': str(file_path),
                        'file_size': stat.st_size,
                        'file_extension': file_path.suffix.lower(),
                        'modified_time': stat.st_mtime
                    })
        
        return sorted(files, key=lambda x: x['modified_time'], reverse=True)
        
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {str(e)}")
        return []

def create_directory(directory: str) -> bool:
    """Create directory if it doesn't exist"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")
        return False
