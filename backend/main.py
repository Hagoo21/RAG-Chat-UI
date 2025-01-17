from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from typing import List, Dict
import json
from openai import OpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from PIL import Image
import fitz  # PyMuPDF
import io
import base64
from datetime import datetime
from pydantic import BaseModel
from contextlib import asynccontextmanager

class MessageRequest(BaseModel):
    message: str

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "ChatMIM"
COLLECTION_NAME = "Incidents"

# Global variables
mongodb_client = None
openai_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global mongodb_client, openai_client
    try:
        mongodb_client = MongoClient(
            MONGODB_URI,
            server_api=ServerApi('1'),
            maxPoolSize=5,
            minPoolSize=1,
            maxIdleTimeMS=30000,
            retryWrites=True,
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=5000
        )
        # Verify connection
        mongodb_client.admin.command('ping')
        print("Connected to MongoDB!")
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        yield
    except Exception as e:
        print(f"Startup error: {e}")
        raise
    finally:
        # Shutdown
        if mongodb_client:
            mongodb_client.close()
            print("Closed MongoDB connection")

app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://adorable-macaron-2074b9.netlify.app", "https://rag-chat-ui-backend:10000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:])
            break
        
        last_space = text.rfind(' ', start, end)
        if last_space != -1:
            chunks.append(text[start:last_space])
            start = last_space - chunk_overlap
        else:
            chunks.append(text[start:end])
            start = end - chunk_overlap
        
        start = max(start, 0)
    
    return chunks

@app.get("/health")
async def health_check():
    try:
        mongodb_client.admin.command('ping')
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/incidents")
async def get_incidents():
    try:
        collection = mongodb_client[DB_NAME][COLLECTION_NAME]
        incidents = list(collection.find({}, {'_id': 0}).limit(50))  # Limit results
        return incidents or []
    except Exception as e:
        print(f"Error fetching incidents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/upload")
async def upload_documents(files: List[UploadFile]):
    try:
        collection = mongodb_client[DB_NAME][COLLECTION_NAME]
        uploaded_count = 0
        
        for file in files:
            content = await file.read()
            file_extension = file.filename.lower().split('.')[-1]
            preview_image = None
            
            # Process different file types
            if file_extension == 'pdf':
                pdf_document = fitz.open(stream=content, filetype="pdf")
                text = ""
                for page in pdf_document:
                    text += page.get_text()
                
                if len(pdf_document) > 0:
                    first_page = pdf_document[0]
                    pix = first_page.get_pixmap(matrix=fitz.Matrix(1, 1))
                    img_data = pix.tobytes("png")
                    preview_image = base64.b64encode(img_data).decode()
                pdf_document.close()
            
            elif file_extension in ['txt', 'csv', 'json']:
                text = content.decode('utf-8', errors='ignore')
            
            elif file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                img = Image.open(io.BytesIO(content))
                img.thumbnail((200, 200))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                preview_image = base64.b64encode(img_byte_arr.getvalue()).decode()
                text = f"Image file: {file.filename}"
            
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
            
            # Create and process chunks
            chunks = create_text_chunks(text)
            embedded_chunks = []
            
            for chunk in chunks:
                try:
                    embedding_response = openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=chunk
                    )
                    embedded_chunks.append({
                        "text": chunk,
                        "embedding": embedding_response.data[0].embedding,
                        "metadata": {
                            "filename": file.filename,
                            "preview_image": preview_image,
                            "file_type": file_extension,
                            "upload_timestamp": datetime.utcnow().isoformat(),
                            "file_size": len(content)
                        }
                    })
                except Exception as e:
                    print(f"Error creating embedding: {str(e)}")
                    continue
            
            if embedded_chunks:
                collection.insert_many(embedded_chunks)
                uploaded_count += 1
        
        return {
            "message": f"Successfully processed {uploaded_count} documents",
            "status": "success"
        }
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=Dict[str, str])
async def search_context(request: MessageRequest):
    try:
        collection = mongodb_client[DB_NAME][COLLECTION_NAME]
        
        # Get embedding for the query
        query_embedding_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=request.message
        )
        query_embedding = query_embedding_response.data[0].embedding

        # Perform vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_search_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 5
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        contexts = [doc["text"] for doc in results]
        concatenated_context = " ".join(contexts)
        
        return {"context": concatenated_context}
    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
