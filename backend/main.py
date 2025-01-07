from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from typing import List
import json
from openai import OpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Atlas connection
db_client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))
DB_NAME = "ChatMIM"
COLLECTION_NAME = "Incidents"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "Vector-Search"
MONGODB_COLLECTION = db_client[DB_NAME][COLLECTION_NAME]

# OpenAI setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Vector store setup
vectorstore = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

@app.post("/upload")
async def upload_documents(files: List[UploadFile]):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )
        
        for file in files:
            content = await file.read()
            text = content.decode()
            
            # Create document chunks
            chunks = text_splitter.split_text(text)
            documents = [Document(page_content=chunk, metadata={"filename": file.filename}) for chunk in chunks]
            
            # Create embeddings and store in MongoDB
            embedded_chunks = [{
                "content": doc.page_content,
                "embedding": embedding_model.embed_query(doc.page_content),
                "metadata": doc.metadata
            } for doc in documents]
            
            MONGODB_COLLECTION.insert_many(embedded_chunks)
            
        return {"message": "Documents uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: dict):
    try:
        user_input = request.get("message")
        if not user_input:
            raise HTTPException(status_code=400, detail="Message is required")

        # Get relevant documents
        retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})
        relevant_docs = retriever.get_relevant_documents(user_input)
        context = ". ".join([doc.page_content for doc in relevant_docs])

        # Create chat completion
        system_message = """You are an assistant whose work is to review the context data and provide appropriate answers from the context. 
        Answer only using the context provided. If the answer is not found in the context, respond "I don't know"."""
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
            ],
            temperature=0
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "context": [doc.page_content for doc in relevant_docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))