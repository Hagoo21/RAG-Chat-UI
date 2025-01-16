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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/incidents")
async def get_incidents():
    try:
        print("Attempting to connect to database...")
        # First verify connection
        db_client.admin.command('ping')
        print("Successfully connected to database")
        
        print(f"Attempting to fetch incidents from {DB_NAME}.{COLLECTION_NAME}")
        # Fetch all documents from the collection
        incidents = list(MONGODB_COLLECTION.find({}, {'_id': 0}))
        print(f"Found {len(incidents)} incidents")
        if not incidents:
            return []
        return incidents
    except Exception as e:
        print(f"Error fetching incidents: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Database error: {str(e)}"
        )

@app.post("/upload")
async def upload_documents(files: List[UploadFile]):
    try:
        uploaded_docs = []
        
        for file in files:
            content = await file.read()
            file_extension = file.filename.lower().split('.')[-1]
            
            # Generate preview image
            preview_image = None
            text = ""
            
            if file_extension == 'pdf':
                # Handle PDF files
                pdf_document = fitz.open(stream=content, filetype="pdf")
                # Extract text from PDF
                for page in pdf_document:
                    text += page.get_text()
                
                # Generate preview from first page
                if len(pdf_document) > 0:
                    first_page = pdf_document[0]
                    pix = first_page.get_pixmap(matrix=fitz.Matrix(1, 1))
                    img_data = pix.tobytes("png")
                    preview_image = base64.b64encode(img_data).decode()
                pdf_document.close()
            
            elif file_extension in ['txt', 'csv', 'json']:
                # Handle text files
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    text = content.decode('latin-1')
            
            elif file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                # Handle image files
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
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_extension}"
                )

            # Create document
            document = {
                "text": text,
                "metadata": {
                    "filename": file.filename,
                    "preview_image": preview_image,
                    "file_type": file_extension,
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "file_size": len(content)
                }
            }
            
            # Store in MongoDB
            MONGODB_COLLECTION.insert_one(document)
            uploaded_docs.append(document)
            
        return {"message": "Documents uploaded and processed successfully"}
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def search_context(request: MessageRequest):
    """
    Searches for relevant contexts using text similarity.
    """
    try:
        # Simple text search in MongoDB
        cursor = MONGODB_COLLECTION.find(
            {"$text": {"$search": request.message}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(5)
        
        contexts = [doc.get("text", "") for doc in cursor]
        concatenated_context = " ".join(contexts) if contexts else "No relevant context found."
        
        return {"context": concatenated_context}
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# @app.post("/chat")
# async def chat_endpoint(request: dict):
#     try:
#         user_input = request.get("message")
#         if not user_input:
#             raise HTTPException(status_code=400, detail="Message is required")

#         # Get relevant documents
#         retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})
#         relevant_docs = retriever.get_relevant_documents(user_input)
#         context = ". ".join([doc.page_content for doc in relevant_docs])

#         # Create chat completion
#         system_message = """You are an assistant whose work is to review the context data and provide appropriate answers from the context. 
#         Answer only using the context provided. If the answer is not found in the context, respond "I don't know"."""
        
#         response = openai_client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
#             ],
#             temperature=0
#         )

#         return {
#             "answer": response.choices[0].message.content.strip(),
#             "context": [doc.page_content for doc in relevant_docs]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))