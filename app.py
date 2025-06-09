import streamlit as st
import json
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import tempfile
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Talk to YourAPI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .document-info {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class JSONRAGSystem:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            st.error("Please set your GOOGLE_API_KEY in the .env file")
            return
        
        # Initialize embeddings and LLM
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.google_api_key,
                temperature=0.3
            )
        except Exception as e:
            st.error(f"Error initializing Google AI services: {str(e)}")
            return
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vectorstore = None
        self.qa_chain = None
        
    def flatten_json(self, data: Any, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested JSON structure"""
        items = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(self.flatten_json(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        items.extend(self.flatten_json(item, f"{new_key}[{i}]", sep=sep).items())
                else:
                    items.append((new_key, str(v)))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                items.extend(self.flatten_json(item, f"{parent_key}[{i}]", sep=sep).items())
        else:
            items.append((parent_key, str(data)))
        return dict(items)
    
    def json_to_text(self, json_data: Dict[str, Any], filename: str) -> str:
        """Convert JSON data to readable text format"""
        flattened = self.flatten_json(json_data)
        text_parts = [f"Document: {filename}\n"]
        
        for key, value in flattened.items():
            text_parts.append(f"{key}: {value}")
        
        return "\n".join(text_parts)
    
    def process_json_files(self, uploaded_files: List) -> List[Document]:
        """Process uploaded JSON files and create documents"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Read and parse JSON
                json_content = json.loads(uploaded_file.read())
                
                # Convert JSON to text
                text_content = self.json_to_text(json_content, uploaded_file.name)
                
                # Create document with metadata
                doc = Document(
                    page_content=text_content,
                    metadata={
                        'filename': uploaded_file.name,
                        'source': 'uploaded_json',
                        'content_type': 'json'
                    }
                )
                documents.append(doc)
                
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON file {uploaded_file.name}: {str(e)}")
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        
        return documents
    
    def create_vectorstore(self, documents: List[Document]):
        """Create FAISS vectorstore from documents"""
        if not documents:
            st.error("No valid documents to process")
            return
        
        try:
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.qa_chain:
            return {"error": "No documents loaded. Please upload JSON files first."}
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 JSON RAG System</h1>
        <p>Upload multiple JSON documents and ask questions using AI-powered search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = JSONRAGSystem()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize document info
    if 'loaded_documents' not in st.session_state:
        st.session_state.loaded_documents = []
    
    # Sidebar for file upload and document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload JSON Files",
            type=['json'],
            accept_multiple_files=True,
            help="Select one or more JSON files to upload"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    documents = st.session_state.rag_system.process_json_files(uploaded_files)
                    
                    if documents:
                        success = st.session_state.rag_system.create_vectorstore(documents)
                        
                        if success:
                            st.session_state.loaded_documents = [doc.metadata['filename'] for doc in documents]
                            st.success(f"✅ Successfully processed {len(documents)} documents!")
                        else:
                            st.error("❌ Failed to process documents")
                    else:
                        st.error("❌ No valid documents found")
        
        # Document information
        if st.session_state.loaded_documents:
            st.subheader("📋 Loaded Documents")
            for i, doc_name in enumerate(st.session_state.loaded_documents, 1):
                st.markdown(f"**{i}.** {doc_name}")
        
        # Clear documents
        if st.session_state.loaded_documents:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.loaded_documents = []
                st.session_state.chat_history = []
                st.session_state.rag_system = JSONRAGSystem()
                st.success("Documents cleared!")
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**🙋 You:** {question}")
                st.markdown(f"**🤖 Assistant:** {answer}")
                st.divider()
        
        # Query input
        if st.session_state.loaded_documents:
            question = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What are the main topics in the uploaded documents?",
                key="question_input"
            )
            
            col_ask, col_clear = st.columns([3, 1])
            
            with col_ask:
                if st.button("🔍 Ask Question", type="primary", disabled=not question):
                    if question:
                        with st.spinner("Searching for answer..."):
                            result = st.session_state.rag_system.query(question)
                            
                            if "error" in result:
                                st.error(result["error"])
                            else:
                                st.session_state.chat_history.append((question, result["answer"]))
                                st.rerun()
            
            with col_clear:
                if st.button("🧹 Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        else:
            st.info("👆 Please upload and process JSON files first to start asking questions.")
    
    with col2:
        st.header("ℹ️ System Information")
        
        # System status
        st.subheader("🔧 Status")
        if st.session_state.rag_system.google_api_key:
            st.success("✅ Google API Connected")
        else:
            st.error("❌ Google API Not Connected")
        
        if st.session_state.loaded_documents:
            st.success(f"✅ {len(st.session_state.loaded_documents)} Documents Loaded")
        else:
            st.warning("⚠️ No Documents Loaded")
        
        # Usage tips
        st.subheader("💡 Usage Tips")
        st.markdown("""
        1. **Upload JSON Files**: Use the sidebar to upload multiple JSON files
        2. **Process Documents**: Click 'Process Documents' to index your files
        3. **Ask Questions**: Type questions about your data in the chat interface
        4. **Natural Language**: Ask questions in plain English
        5. **Multiple Files**: The system can search across all uploaded documents
        """)
        
        # Example questions
        if st.session_state.loaded_documents:
            st.subheader("❓ Example Questions")
            example_questions = [
                "What are the main topics in these documents?",
                "Summarize the key information",
                "What data fields are available?",
                "Show me specific values for [field name]",
                "Compare information across documents"
            ]
            
            for eq in example_questions:
                if st.button(eq, key=f"example_{eq[:20]}"):
                    st.session_state.question_input = eq

if __name__ == "__main__":
    main()