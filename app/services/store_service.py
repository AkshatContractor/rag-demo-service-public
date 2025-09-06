import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

def ingest_documents():
    data_dir = "app\data" 
    document_list_path = os.path.join(data_dir, "documents_to_ingest.txt")
    
    docs = []
    
    try:
        with open(document_list_path, 'r', encoding='utf-8') as f:
            document_names = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: The file {document_list_path} was not found.")
        return 0
        
    for doc_name in document_names:
        if not doc_name or doc_name.startswith('#'):
            continue  # Skip empty lines or comments
            
        file_path = os.path.join(data_dir, doc_name.strip())
        
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        vector_store = QdrantVectorStore.from_documents(
            chunks,
            embedding=embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="test-col-1",
            force_recreate=True
        )
    except Exception as e:
        print(f"Failed to store documents in Qdrant: {e}")
        return 0

    print(f"Successfully ingested {len(chunks)} chunks into Qdrant.")
    return len(chunks)