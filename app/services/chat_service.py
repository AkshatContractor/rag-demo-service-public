import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fastapi import HTTPException, status

load_dotenv()

#  Configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
TOP_K = 6
DEBUG = False 

#  Qdrant client 
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Embeddings 
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

# LLM 
llm = ChatGoogleGenerativeAI(
    model=os.getenv("LANGUAGE_MODEL"),
    temperature=0.2,
    top_p=0.9,
    max_output_tokens=300,
    verbose=False
)

# Prompt Loading
try:
    with open("prompt.txt", "r", encoding="utf-8") as f:
        template_str = f.read()
except FileNotFoundError:
    template_str = os.getenv("PROMPT_TEMPLATE")

if not template_str:
    template_str = """You are a helpful AI assistant. Use ONLY the provided context.
If the context lacks the answer, say you do not have enough information.

Context:
{context}

Question: {question}

Answer (concise, direct):"""

if "{query}" in template_str and "{question}" not in template_str:
    template_str = template_str.replace("{query}", "{question}")

prompt = PromptTemplate(
    template=template_str,
    input_variables=["question", "context"]
)

# Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K}),
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def _fallback_summarize(docs, original_query: str) -> str:
    """Attempt a second-chance answer if primary chain returned blank."""
    if not docs:
        return ""
    joined = "\n\n---\n\n".join(d.page_content[:1200] for d in docs[:5])
    fallback_prompt = f"""You are repairing an empty answer for a user query.

User Question: {original_query}

Below are retrieved context snippets. Create a factual, well-structured answer.
If nothing is relevant, say you do not have enough information.

Context:
{joined}

Answer:"""
    try:
        resp = llm.invoke(fallback_prompt)
        return (resp.content if hasattr(resp, "content") else str(resp)).strip()
    except Exception:
        return ""

def ask_question(query: str):
    try:
        # Primary invoke
        result = qa_chain.invoke({"query": query})
        answer = (result.get("result") or "").strip()
        source_docs = result.get("source_documents", []) or []

        # Fallback if blank
        if not answer:
            fallback_answer = _fallback_summarize(source_docs, query).strip()
            if fallback_answer:
                answer = fallback_answer

        if not answer:
            return {
                "answer": "I do not have enough information to answer that.",
                "sources": []
            }

        sources = [doc.metadata for doc in source_docs]

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the query: {str(e)}"
        )
