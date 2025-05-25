#!/usr/bin/env python3
"""
RAG System Startup Script
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_ollama_model(model_name="llama2"):
    """Check if Ollama model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(model_name in model.get("name", "") for model in models)
        return False
    except:
        return False

def start_ollama():
    """Start Ollama service"""
    print("🚀 Starting Ollama service...")
    try:
        # Try to start Ollama in background
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for Ollama to start
        for i in range(30):
            if check_ollama_running():
                print("✅ Ollama service started successfully")
                return True
            time.sleep(1)
            print(f"⏳ Waiting for Ollama to start... ({i+1}/30)")
        
        print("❌ Failed to start Ollama service")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first:")
        print("   macOS: brew install ollama")
        print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        return False

def pull_model(model_name="qwen3:4b"):
    """Pull Ollama model if not available"""
    if not check_ollama_model(model_name):
        print(f"📥 Pulling {model_name} model...")
        try:
            result = subprocess.run(["ollama", "pull", model_name], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {model_name} model downloaded successfully")
                return True
            else:
                print(f"❌ Failed to download {model_name} model: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    else:
        print(f"✅ {model_name} model already available")
        return True

def setup_environment():
    """Setup environment and directories"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = ["data/uploads", "data/vector_db", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Check if .env exists
    if not Path(".env").exists():
        print("⚠️  .env file not found. Using default configuration.")
        print("   You can create a .env file to customize settings.")
    
    print("✅ Environment setup complete")

def start_backend():
    """Start backend service"""
    print("🚀 Starting backend service...")
    
    # Change to backend directory
    backend_dir = Path("backend")
    print(f"   Backend directory: {backend_dir.resolve()}")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return None

    try:
        # Start backend service
        process = subprocess.Popen([
            sys.executable, "-m", "backend.app.main"
        ])
        
        # Wait for backend to start
        for i in range(30):
            try:
                response = requests.get("http://localhost:8000/api/v1/status/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend service started successfully")
                    print("   API Documentation: http://localhost:8000/docs")
                    return process
            except:
                pass
            time.sleep(1)
            print(f"⏳ Waiting for backend to start... ({i+1}/30)")
        
        print("❌ Backend service failed to start properly")
        return process
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start frontend service"""
    print("🚀 Starting frontend service...")
    
    # Change to frontend directory
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None
    
    try:
        # Start frontend service
        process = subprocess.Popen([
            "streamlit", "run", "streamlit_app.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], cwd=str(frontend_dir))
        
        # Give frontend time to start
        time.sleep(5)
        print("✅ Frontend service started successfully")
        print("   Web Interface: http://localhost:8501")
        return process
        
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("=" * 50)
    print("🤖 RAG智能问答系统启动脚本")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check and start Ollama
    if not check_ollama_running():
        if not start_ollama():
            print("❌ Cannot start system without Ollama")
            return
    else:
        print("✅ Ollama service is already running")
    
    # Pull model if needed
    if not pull_model():
        print("⚠️  Model not available, but continuing...")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Cannot start system without backend")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Frontend failed to start")
        if backend_process:
            backend_process.terminate()
        return
    
    print("\n" + "=" * 50)
    print("🎉 RAG系统启动成功!")
    print("=" * 50)
    print("📊 服务地址:")
    print("   前端界面: http://localhost:8501")
    print("   后端API: http://localhost:8000")
    print("   API文档: http://localhost:8000/docs")
    print("\n💡 使用说明:")
    print("   1. 打开前端界面上传文档")
    print("   2. 在智能问答页面提问")
    print("   3. 在系统监控页面查看状态")
    print("\n⚠️  按 Ctrl+C 停止所有服务")
    print("=" * 50)
    
    try:
        # Keep processes running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务...")
        if frontend_process:
            frontend_process.terminate()
        if backend_process:
            backend_process.terminate()
        print("✅ 所有服务已停止")

if __name__ == "__main__":
    main()
