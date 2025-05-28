import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load .env file
load_dotenv()

def verify_source_deletion(vectorstore, sources_to_delete):
    try:
        all_docs = vectorstore.get()

        if not all_docs or 'metadatas' not in all_docs:
            return {
                "success": True,
                "remaining_sources": [],
                "total_docs": 0
            }

        existing_sources = {doc['source'] for doc in all_docs['metadatas']}

        remaining_sources = [source for source in sources_to_delete if source in existing_sources]

        return {
            "success": len(remaining_sources) == 0,
            "remaining_sources": remaining_sources,
            "total_docs": len(existing_sources)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "remaining_sources": sources_to_delete,
            "total_docs": 0
        }


def load_vectorstore(uploaded_files=None, urls=None):
    """Load documents into ChromaDB vectorstore and return document summaries.
    
    Args:
        uploaded_files: List of uploaded file objects
        urls: List of URLs to process
        
    Returns:
        tuple: (vectorstore, document_summaries)
    """
    all_chunks = []
    document_summaries = {}
    
    # Process PDF files
    if uploaded_files:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(file.read())
                temp_path = temp.name
            
            try:
                docs = PyPDFLoader(temp_path).load()
                num_pages = len(docs)  # Get total number of pages
                
                # Add basic metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": file.name,
                        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "document_type": "pdf",
                        "num_pages": num_pages
                    })
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=500
                )
                chunks = text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                
                # Add to document summaries
                document_summaries[file.name] = {
                    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "document_type": "pdf",
                    "num_pages": num_pages 
                }
            finally:
                os.remove(temp_path)
    
    # Process URLs
    if urls:
        for url in urls:
            try:
                docs = WebBaseLoader(url).load()
                # Add basic metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": url,
                        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "document_type": "webpage"
                    })
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=500
                )
                chunks = text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                
                # Add to document summaries
                document_summaries[url] = {
                    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "document_type": "webpage"
                }
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
    
    # Initialize vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings,
                collection_name="joint_planning"
            )
            
            # If no new documents to add, just return existing vectorstore
            if not all_chunks:
                # Get existing documents and create summaries
                existing_docs = vectorstore.get()
                if existing_docs and 'metadatas' in existing_docs:
                    for metadata in existing_docs['metadatas']:
                        source = metadata['source']
                        if source not in document_summaries:
                            document_summaries[source] = {
                                "upload_time": metadata.get('upload_time', ''),
                                "document_type": metadata.get('document_type', ''),
                                "num_pages": metadata.get('num_pages', 0)
                            }
                    return vectorstore, document_summaries
            
            # Add new chunks if there are any
            existing_sources = {doc['source'] for doc in vectorstore.get().get('metadatas', [])}
            new_chunks = [chunk for chunk in all_chunks if chunk.metadata['source'] not in existing_sources]
            
            if new_chunks:
                vectorstore.add_documents(new_chunks)
        else:
            vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory="./chroma_db",
                collection_name="joint_planning"
            )
    except Exception as e:
        print(f"Error initializing vectorstore: {str(e)}")
        # If there's an error, try to create a fresh vectorstore
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name="joint_planning"
        )
    
    return vectorstore, document_summaries