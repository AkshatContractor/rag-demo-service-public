import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

def ingest_documents():
    data_dir = "app/data"
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
            continue
        file_path = os.path.join(data_dir, doc_name.strip())
        try:
            loader = PyPDFLoader(file_path) if file_path.endswith('.pdf') else TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            
            for d in loaded_docs:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                d.metadata["filename"] = doc_name.strip()
                d.metadata["category"] = base_name.split("_")[0] if "_" in base_name else base_name
                if "projects" in doc_name.lower() or "project" in doc_name.lower():
                    d.metadata["owner"] = "Akshat Contractor"
                if not d.page_content.startswith("# Section"):
                    if "projects" in doc_name.lower():
                        d.page_content = f"# Section: projects by Akshat Contractor\n{d.page_content}"
                    else:
                        d.page_content = f"# Section: {doc_name}\n{d.page_content}"
            
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    project_chunks = [chunk for chunk in chunks if "projects" in chunk.metadata.get("filename", "").lower()]
    if project_chunks:
        summary_content = "# Section: Summary of Akshat's Projects\n"
        for chunk in project_chunks:
            summary_content += chunk.page_content + "\n\n"
        
        summary_doc = project_chunks[0].copy()
        summary_doc.page_content = summary_content
        summary_doc.metadata["summary"] = True
        chunks.append(summary_doc)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        vector_store = QdrantVectorStore.from_documents(
            chunks,
            embedding=embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME_INGEST"),
            force_recreate=True
        )
    except Exception as e:
        print(f"Failed to store documents in Qdrant: {e}")
        return 0

    print(f"Successfully ingested {len(chunks)} chunks into Qdrant.")
    return len(chunks)
