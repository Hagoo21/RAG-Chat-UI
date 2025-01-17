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

class MessageRequest(BaseModel):
    message: str

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://adorable-macaron-2074b9.netlify.app", "https://rag-chat-ui-backend:10000","http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Atlas connection
db_client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))
DB_NAME = "ChatMIM"
COLLECTION_NAME = "Incidents"
MONGODB_COLLECTION = db_client[DB_NAME][COLLECTION_NAME]

# OpenAI setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        # Handle the last chunk
        if end >= text_length:
            chunks.append(text[start:])
            break
        
        # Find the last space before the end to avoid cutting words
        last_space = text.rfind(' ', start, end)
        if last_space != -1:
            chunks.append(text[start:last_space])
            start = last_space - chunk_overlap
        else:
            chunks.append(text[start:end])
            start = end - chunk_overlap
        
        start = max(start, 0)  # Ensure start doesn't go negative
    
    return chunks

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/incidents")
async def get_incidents():
    try:
        print("Attempting to connect to database...")
        db_client.admin.command('ping')
        print("Successfully connected to database")
        
        print(f"Attempting to fetch incidents from {DB_NAME}.{COLLECTION_NAME}")
        incidents = list(MONGODB_COLLECTION.find({}, {'_id': 0}))
        print(f"Found {len(incidents)} incidents")
        return incidents or []
    except Exception as e:
        print(f"Error fetching incidents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/upload")
async def upload_documents(files: List[UploadFile]):
    try:
        for file in files:
            content = await file.read()
            file_extension = file.filename.lower().split('.')[-1]
            preview_image = None
            
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
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    text = content.decode('latin-1')
            
            elif file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                try:
                    img = Image.open(io.BytesIO(content))
                    img.thumbnail((200, 200))
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    preview_image = base64.b64encode(img_byte_arr).decode()
                    text = f"Image file: {file.filename}"
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    text = f"Image file: {file.filename}"
            
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
            
            # Create document chunks
            chunks = create_text_chunks(text)
            
            # Create embeddings using OpenAI directly
            embedded_chunks = []
            for chunk in chunks:
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
            
            if embedded_chunks:
                MONGODB_COLLECTION.insert_many(embedded_chunks)
            
        return {"message": "Documents uploaded and processed successfully"}
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=Dict[str, str])
async def search_context(request: MessageRequest):
    """
    Searches for relevant contexts using MongoDB Atlas Vector Search.
    """
    try:
        # Get embedding for the query
        query_embedding_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=request.message
        )
        query_embedding = query_embedding_response.data[0].embedding

        # Perform vector search using MongoDB's $vectorSearch
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_search_index",  # Make sure this matches your index name
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 5
                }
            }
        ]

        results = list(MONGODB_COLLECTION.aggregate(pipeline))
        contexts = [doc["text"] for doc in results]
        concatenated_context = " ".join(contexts)
        
        return {"context": concatenated_context}
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))