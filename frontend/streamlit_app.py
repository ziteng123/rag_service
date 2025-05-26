"""
Streamlit frontend for RAG system
"""

import streamlit as st
import requests
import json
import time
from typing import  Dict, Any
import pandas as pd
import plotly.express as px

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Page configuration
st.set_page_config(
    page_title="RAG问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .document-item {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/status/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def upload_files(files) -> Dict[str, Any]:
    """Upload files to the API"""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.getvalue(), file.type)))
        
        response = requests.post(f"{API_BASE_URL}/upload", files=files_data, timeout=300)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_uploaded_documents() -> Dict[str, Any]:
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/upload/documents", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "documents": []}
    except Exception as e:
        return {"status": "error", "documents": [], "error": str(e)}

def delete_document(filename: str) -> Dict[str, Any]:
    """Delete a specific document"""
    try:
        response = requests.delete(f"{API_BASE_URL}/upload/documents/{filename}", timeout=30)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"Failed to delete {filename}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def delete_all_documents() -> Dict[str, Any]:
    """Delete all documents"""
    try:
        payload = {"delete_all": True}
        response = requests.post(f"{API_BASE_URL}/upload/documents/delete", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": "Failed to delete all documents"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def query_documents_stream(question: str, top_k: int = 5, status_placeholder=None):
    """Query documents with streaming response"""
    try:
        status_placeholder.info("🔍 正在检索相关文档...")
        payload = {
            "question": question,
            "top_k": top_k,
            "model": st.session_state.get("selected_model", "qwen3:4b")
        }
        response = requests.post(f"{API_BASE_URL}/query/stream", json=payload, timeout=60, stream=True)
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    yield data
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        yield {"type": "error", "error": str(e)}

def get_models() -> Dict[str, Any]:
    """Get available models"""
    try:
        response = requests.get(f"{API_BASE_URL}/query/models", timeout=30)
        return response.json() or {}
    except Exception as e:
        st.error(f"Error getting models: {str(e)}")
        return {}

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def format_upload_time(upload_time: str) -> str:
    """Format upload time"""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return upload_time

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG智能问答系统</h1>', unsafe_allow_html=True)
    
    # Check API health 
    if not check_api_health():
        st.error("⚠️ 无法连接到后端API服务，请确保服务正在运行")
        st.info("请运行: `cd backend && python -m backend.app.main`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 系统状态")
        
        # Get system status
        status = get_system_status()
        if status:
            st.success(f"状态: {status.get('status', 'unknown')}")
            st.info(f"文档数量: {status.get('total_documents', 0)}")
            st.info(f"文档块数量: {status.get('total_chunks', 0)}")
            st.info(f"运行时间: {status.get('uptime', 'unknown')}")
            
            # Ollama status
            if status.get('ollama_available'):
                st.success("✅ Ollama 可用")
            else:
                st.error("❌ Ollama 不可用")
            
            # ChromaDB status
            if status.get('chromadb_available'):
                st.success("✅ ChromaDB 可用")
            else:
                st.error("❌ ChromaDB 不可用")
        else:
            st.error("无法获取系统状态")
        
        st.divider()
        # Models
        st.header("🧠 模型选择")
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "qwen3:4b"
        model_list = get_models().get("models", [])
        if model_list:
            st.session_state.selected_model = st.selectbox(
                "选择后端模型",
                model_list,
                index=model_list.index(st.session_state.selected_model)
                if st.session_state.selected_model in model_list
                else 0
            )
        else:
            st.warning("没有可用的模型，请检查后端配置")
        st.markdown(f"当前模型: `{st.session_state.selected_model}`")
        st.divider()
        # Settings
        st.header("⚙️ 设置")
        top_k = st.slider("检索文档数量", min_value=1, max_value=20, value=5)
        
        # Clear chat history
        if st.button("🗑️ 清空对话历史"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize current tab
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "💬 智能问答"
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 智能问答", "📁 文档管理", "📈 系统监控"])
    
    with tab1:
        st.header("💬 智能问答")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display sources if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 参考来源"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>来源 {i+1}:</strong> {source.get('filename', 'unknown')}<br>
                                <strong>相似度:</strong> {source.get('similarity_score', 0):.3f}<br>
                                <strong>内容预览:</strong> {source.get('content_preview', '')[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📁 文档管理")
        
        # Upload section
        with st.container():
            st.subheader("📤 上传新文档")
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            
            # File uploader
            uploaded_files = st.file_uploader(
                "选择要上传的文档",
                type=['pdf', 'docx', 'pptx', 'txt', 'md'],
                accept_multiple_files=True,
                help="支持的格式: PDF, Word, PowerPoint, 文本文件, Markdown"
            )
            
            if uploaded_files:
                st.info(f"已选择 {len(uploaded_files)} 个文件")
                
                # Display selected files
                for file in uploaded_files:
                    st.write(f"📄 {file.name} ({format_file_size(file.size)})")
                
                # Upload button
                if st.button("🚀 开始上传和处理", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("正在上传文件...")
                    progress_bar.progress(25)
                    
                    # Upload files
                    result = upload_files(uploaded_files)
                    
                    progress_bar.progress(100)
                    
                    if result.get("status") == "success":
                        st.success(f"✅ 成功处理 {result.get('file_count', 0)} 个文件")
                        st.info(f"总共生成 {result.get('total_chunks', 0)} 个文档块")
                        
                        if result.get("failed_files"):
                            st.warning(f"以下文件处理失败: {', '.join(result['failed_files'])}")
                        
                        # Refresh the page to show new documents
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ 上传失败: {result.get('message', '未知错误')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Document list section
        st.subheader("📋 已上传的文档")
        
        # Refresh and delete all buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 刷新列表"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ 删除全部", type="secondary"):
                if st.session_state.get("confirm_delete_all", False):
                    with st.spinner("正在删除所有文档..."):
                        result = delete_all_documents()
                        if result.get("status") == "success":
                            st.success("✅ 已删除所有文档")
                            st.session_state["confirm_delete_all"] = False
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ 删除失败: {result.get('message', '未知错误')}")
                else:
                    st.session_state["confirm_delete_all"] = True
                    st.warning("⚠️ 再次点击确认删除所有文档")
        
        # Get and display documents
        documents_result = get_uploaded_documents()
        
        if documents_result.get("status") == "success":
            documents = documents_result.get("documents", [])
            
            if documents:
                st.info(f"共有 {len(documents)} 个文档")
                
                # Display documents
                for i, doc in enumerate(documents):
                    with st.container():
                        st.markdown(f'<div class="document-item">', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Document info
                            st.markdown(f"**📄 {doc.get('filename', 'Unknown')}**")
                            
                            # Document details
                            col_info1, col_info2, col_info3 = st.columns(3)
                            with col_info1:
                                st.caption(f"类型: {doc.get('file_type', 'unknown')}")
                            with col_info2:
                                st.caption(f"大小: {format_file_size(doc.get('file_size', 0))}")
                            with col_info3:
                                st.caption(f"块数: {doc.get('chunk_count', 0)}")
                            
                            # Upload time and metadata
                            upload_time = format_upload_time(doc.get('upload_time', ''))
                            st.caption(f"上传时间: {upload_time}")
                            
                            if doc.get('title'):
                                st.caption(f"标题: {doc.get('title')}")
                            if doc.get('author'):
                                st.caption(f"作者: {doc.get('author')}")
                        
                        with col2:
                            # Delete button
                            delete_key = f"delete_{i}_{doc.get('filename', '')}"
                            if st.button("🗑️ 删除", key=delete_key, type="secondary"):
                                with st.spinner(f"正在删除 {doc.get('filename', '')}..."):
                                    result = delete_document(doc.get('filename', ''))
                                    if result.get("status") == "success":
                                        st.success(f"✅ 已删除文档: {doc.get('filename', '')}")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ 删除失败: {result.get('message', '未知错误')}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("")  # Add some spacing
            else:
                st.info("📭 暂无已上传的文档")
                st.markdown("请使用上方的文件上传功能添加文档。")
        else:
            st.error(f"❌ 获取文档列表失败: {documents_result.get('error', '未知错误')}")
    
    with tab3:
        st.header("📈 系统监控")
        
        # Refresh button
        if st.button("🔄 刷新状态"):
            st.rerun()
        
        # Get detailed status
        try:
            response = requests.get(f"{API_BASE_URL}/status/detailed", timeout=10)
            if response.status_code == 200:
                detailed_status = response.json()
                
                # System metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "CPU使用率", 
                        f"{detailed_status.get('system_metrics', {}).get('cpu_percent', 0):.1f}%"
                    )
                
                with col2:
                    memory = detailed_status.get('system_metrics', {}).get('memory', {})
                    st.metric(
                        "内存使用率", 
                        f"{memory.get('used_percent', 0):.1f}%",
                        f"{memory.get('available_gb', 0):.1f}GB 可用"
                    )
                
                with col3:
                    disk = detailed_status.get('system_metrics', {}).get('disk', {})
                    st.metric(
                        "磁盘使用率", 
                        f"{disk.get('used_percent', 0):.1f}%",
                        f"{disk.get('free_gb', 0):.1f}GB 可用"
                    )
                
                # Component status
                st.subheader("组件状态")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    ollama_status = detailed_status.get('ollama', {})
                    if ollama_status.get('status') == 'healthy':
                        st.success(f"✅ Ollama ({ollama_status.get('model', 'unknown')})")
                    else:
                        st.error("❌ Ollama")
                
                with col2:
                    vector_status = detailed_status.get('vector_store', {})
                    if vector_status.get('status') == 'healthy':
                        st.success(f"✅ ChromaDB ({vector_status.get('total_documents', 0)} 文档)")
                    else:
                        st.error("❌ ChromaDB")
                
                # File types chart
                file_types = vector_status.get('file_types', {})
                if file_types:
                    st.subheader("文档类型分布")
                    df = pd.DataFrame(list(file_types.items()), columns=['文件类型', '数量'])
                    fig = px.pie(df, values='数量', names='文件类型', title="文档类型分布")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Configuration
                st.subheader("系统配置")
                config = detailed_status.get('configuration', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"文档块大小: {config.get('chunk_size', 0)}")
                    st.info(f"块重叠: {config.get('chunk_overlap', 0)}")
                
                with col2:
                    st.info(f"最大文件大小: {config.get('max_file_size_mb', 0)}MB")
                    st.info(f"检索数量: {config.get('retrieval_top_k', 0)}")
                
            else:
                st.error("无法获取详细状态信息")
                
        except Exception as e:
            st.error(f"获取监控信息失败: {str(e)}")
    
    # Chat input (must be outside of tabs/containers)
    if prompt := st.chat_input("请输入您的问题..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Create placeholder for assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Initialize response variables
            response_parts = []
            full_response = ""
            sources = []
            error_occurred = False
            
            # Process streaming response
            try:
                for data in query_documents_stream(prompt, top_k, status_placeholder):
                    if data.get("type") == "status":
                        status_placeholder.info(f"🔍 {data.get('message', '')}")
                    
                    elif data.get("type") == "sources":
                        sources = data.get("sources", [])
                        status_placeholder.info("📚 已找到相关文档，正在生成回答...")
                    
                    elif data.get("type") == "answer":
                        new_text = data.get("answer", "")
                        for token in new_text:
                            response_parts.append(token)
                            full_response = ''.join(response_parts)
                            message_placeholder.markdown(full_response)
                            time.sleep(0.01)  # 控制输出速度
                    
                    elif data.get("type") == "complete":
                        processing_time = data.get("processing_time", 0)
                        retrieved_chunks = data.get("retrieved_chunks", 0)
                        status_placeholder.success(f"✅ 完成 (耗时: {processing_time:.2f}s, 检索块数: {retrieved_chunks})")
                        time.sleep(1)  # Show completion message briefly
                        status_placeholder.empty()
                    
                    elif data.get("type") == "error":
                        error_msg = f"抱歉，处理您的问题时出现错误：{data.get('error', '未知错误')}"
                        message_placeholder.error(error_msg)
                        full_response = error_msg
                        error_occurred = True
                        status_placeholder.empty()
                
                # Add assistant message to chat history
                if not error_occurred:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                    
                    # Display sources if available
                    if sources:
                        with st.expander("📚 参考来源"):
                            for i, source in enumerate(sources):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>来源 {i+1}:</strong> {source.get('filename', 'unknown')}<br>
                                    <strong>相似度:</strong> {source.get('similarity_score', 0):.3f}<br>
                                    <strong>内容预览:</strong> {source.get('content_preview', '')[:200]}...
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
                    
            except Exception as e:
                error_msg = f"抱歉，处理您的问题时出现错误：{str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
                status_placeholder.empty()
        
        # Rerun to update the chat display
        st.rerun()

if __name__ == "__main__":
    main()
