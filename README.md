# RAG智能问答系统

基于LangChain、Ollama和ChromaDB构建的多格式文档RAG（Retrieval-Augmented Generation）系统，支持PDF、Word、PowerPoint、Markdown等多种文档格式的智能问答。

## 系统特性

### 🚀 核心功能
- **多格式文档支持**: PDF、Word、PowerPoint、文本文件、Markdown
- **智能文档解析**: 自动提取文本内容，支持表格和结构化数据
- **向量化存储**: 使用ChromaDB进行高效向量存储和检索
- **智能问答**: 基于Ollama大模型的RAG问答系统
- **实时对话**: 支持上下文感知的多轮对话
- **可视化界面**: 基于Streamlit的友好用户界面

### 🛠 技术架构
- **后端**: FastAPI + LangChain + ChromaDB + Ollama
- **前端**: Streamlit
- **文档解析**: PyPDF2、python-docx、python-pptx等
- **向量数据库**: ChromaDB
- **大语言模型**: Ollama (支持llama2、mistral等)

## 快速开始

### 环境要求
- Python 3.8+
- Ollama (需要预先安装并运行)

### 1. 安装Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# 启动Ollama服务
ollama serve

# 下载模型 (在新终端中运行)
ollama pull llama2
```

### 2. 克隆项目
```bash
git clone <repository-url>
cd rag_service
```

### 3. 安装依赖
```bash
# 安装后端依赖
cd backend
pip install -r requirements.txt

# 安装前端依赖
cd ../frontend
pip install -r requirements.txt
```

### 4. 配置环境
```bash
# 复制环境变量文件
cp .env.example .env

# 编辑配置文件
vim .env
```

### 5. 启动服务

#### 启动后端API服务
```bash
cd backend
python -m backend.app.main
```
后端服务将在 http://localhost:8000 启动

#### 启动前端界面
```bash
cd frontend
streamlit run streamlit_app.py
```
前端界面将在 http://localhost:8501 启动

## 使用指南

### 1. 文档上传
1. 打开前端界面 http://localhost:8501
2. 切换到"文档上传"标签页
3. 选择要上传的文档文件
4. 点击"开始上传和处理"按钮
5. 等待文档处理完成

### 2. 智能问答
1. 切换到"智能问答"标签页
2. 在输入框中输入您的问题
3. 系统将自动检索相关文档并生成回答
4. 可以查看参考来源和相似度评分

### 3. 系统监控
1. 切换到"系统监控"标签页
2. 查看系统状态、性能指标和文档统计
3. 监控Ollama和ChromaDB的运行状态

## API文档

启动后端服务后，可以访问以下地址查看API文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 主要API端点

#### 文档上传
```bash
POST /api/v1/upload
Content-Type: multipart/form-data

# 上传多个文件
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx"
```

#### 智能问答
```bash
POST /api/v1/query
Content-Type: application/json

{
  "question": "您的问题",
  "top_k": 5
}
```

#### 系统状态
```bash
GET /api/v1/status
```

## 配置说明

### 环境变量 (.env)
```bash
# Ollama配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=bge-m3:latest

# ChromaDB配置
CHROMA_PERSIST_DIRECTORY=./data/vector_db

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# 文件上传配置
MAX_FILE_SIZE_MB=50
UPLOAD_DIR=./data/uploads
```

### 系统配置 (backend/config.yaml)
```yaml
# 文档处理配置
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_file_size: 50
  allowed_extensions: [".pdf", ".docx", ".pptx", ".txt", ".md"]

# 检索配置
retrieval:
  top_k: 5
  similarity_threshold: 0.7
```

## 项目结构

```
rag_service/
├── backend/                 # 后端服务
│   ├── app/
│   │   ├── api/            # API路由
│   │   ├── core/           # 核心模块
│   │   ├── models/         # 数据模型
│   │   ├── utils/          # 工具函数
│   │   └── main.py         # 主应用
│   ├── config.yaml         # 配置文件
│   └── requirements.txt    # 依赖列表
├── frontend/               # 前端界面
│   ├── components/         # UI组件
│   ├── streamlit_app.py    # 主应用
│   └── requirements.txt    # 依赖列表
├── data/                   # 数据目录
│   ├── uploads/           # 上传文件
│   └── vector_db/         # 向量数据库
├── .env                   # 环境变量
└── README.md              # 说明文档
```

## 故障排除

### 常见问题

1. **Ollama连接失败**
   - 确保Ollama服务正在运行: `ollama serve`
   - 检查模型是否已下载: `ollama list`
   - 验证端口配置: 默认11434

2. **文档上传失败**
   - 检查文件格式是否支持
   - 确认文件大小不超过限制
   - 查看后端日志获取详细错误信息

3. **问答无响应**
   - 确保已上传相关文档
   - 检查Ollama模型状态
   - 尝试重新表述问题

4. **ChromaDB错误**
   - 检查数据目录权限
   - 清空向量数据库: 删除 `data/vector_db` 目录
   - 重启服务

### 日志查看
```bash
# 查看后端日志
tail -f logs/rag_system.log

# 查看Ollama日志
ollama logs
```

## 开发指南

### 添加新的文档格式支持
1. 在 `backend/app/core/document_parser.py` 中添加解析器
2. 更新 `allowed_extensions` 配置
3. 添加相应的依赖包

### 自定义提示词模板
```bash
# 通过API更新提示词
curl -X POST "http://localhost:8000/api/v1/query/prompt" \
  -H "Content-Type: application/json" \
  -d '{"template": "您的自定义提示词模板"}'
```

### 扩展API功能
1. 在 `backend/app/api/` 目录下创建新的路由文件
2. 在 `main.py` 中注册路由
3. 添加相应的数据模型

## 性能优化

### 建议配置
- **内存**: 建议8GB以上
- **存储**: SSD硬盘，至少10GB可用空间
- **CPU**: 多核处理器，支持向量计算

### 优化建议
1. 调整文档分块大小以平衡检索精度和性能
2. 使用更强大的Ollama模型提升问答质量
3. 定期清理不需要的文档以节省存储空间
4. 考虑使用GPU加速Ollama推理

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com
