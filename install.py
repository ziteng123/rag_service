#!/usr/bin/env python3
"""
RAG System Installation Script
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported")
        print("   Please install Python 3.8 or higher")
        return False

def install_ollama():
    """Install Ollama based on the operating system"""
    system = platform.system().lower()
    
    print("🤖 Installing Ollama...")
    
    if system == "darwin":  # macOS
        if run_command("which brew", "Checking Homebrew", check=False):
            return run_command("brew install ollama", "Installing Ollama via Homebrew")
        else:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
    
    elif system == "linux":
        return run_command("curl -fsSL https://ollama.ai/install.sh | sh", 
                          "Installing Ollama via official installer")
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        print("   Please install Ollama manually from https://ollama.ai/")
        return False

def create_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("🐍 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Determine pip command based on OS
    system = platform.system().lower()
    if system == "windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "data/uploads",
        "data/vector_db", 
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created successfully")
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("⚙️  Creating .env file...")
    
    env_content = """# RAG System Environment Variables

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_EMBEDDING_MODEL=llama2

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./data/vector_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8005

# File Upload Configuration
MAX_FILE_SIZE_MB=50
UPLOAD_DIR=./data/uploads

# Logging
LOG_LEVEL=INFO
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def main():
    """Main installation function"""
    print("=" * 60)
    print("🤖 RAG智能问答系统安装脚本")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install Ollama
    print("\n📋 Step 1: Installing Ollama")
    if not install_ollama():
        print("⚠️  Ollama installation failed, but continuing...")
        print("   You can install Ollama manually later")
    
    # Create virtual environment
    print("\n📋 Step 2: Setting up Python environment")
    if not create_virtual_environment():
        print("❌ Cannot continue without virtual environment")
        return
    
    # Install dependencies
    print("\n📋 Step 3: Installing dependencies")
    if not install_dependencies():
        print("❌ Cannot continue without dependencies")
        return
    
    # Setup directories
    print("\n📋 Step 4: Setting up directories")
    if not setup_directories():
        print("❌ Cannot continue without proper directory structure")
        return
    
    # Create .env file
    print("\n📋 Step 5: Creating configuration")
    if not create_env_file():
        print("⚠️  Configuration file creation failed, but continuing...")
    
    print("\n" + "=" * 60)
    print("🎉 安装完成!")
    print("=" * 60)
    print("📋 下一步:")
    print("   1. 启动Ollama服务: ollama serve")
    print("   2. 下载模型: ollama pull llama2")
    print("   3. 运行系统: python start_system.py")
    print("\n💡 或者直接运行: python start_system.py")
    print("   (启动脚本会自动处理Ollama和模型)")
    print("=" * 60)

if __name__ == "__main__":
    main()
