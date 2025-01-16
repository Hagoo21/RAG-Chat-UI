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
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

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

# CORS middleware - Update to allow specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://adorable-macaron-2074b9.netlify.app", "http://localhost:8080"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Atlas connection
db_client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))
DB_NAME = "ChatMIM"
COLLECTION_NAME = "Incidents"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_search_index"
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )
        
        for file in files:
            content = await file.read()
            file_extension = file.filename.lower().split('.')[-1]
            
            # Generate preview image
            preview_image = None
            
            if file_extension == 'pdf':
                # Handle PDF files
                pdf_document = fitz.open(stream=content, filetype="pdf")
                # Extract text from PDF
                text = ""
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
                preview_image = None  # No preview for text files
            
            elif file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                # Handle image files
                try:
                    img = Image.open(io.BytesIO(content))
                    # Resize image for preview
                    img.thumbnail((200, 200))
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    preview_image = base64.b64encode(img_byte_arr).decode()
                    # For images, use OCR or just store filename as text
                    text = f"Image file: {file.filename}"
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    text = f"Image file: {file.filename}"
            
            else:
                # Unsupported file type
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_extension}"
                )
            
            # Create document chunks
            chunks = text_splitter.split_text(text)
            documents = [Document(
                page_content=chunk, 
                metadata={
                    "filename": file.filename,
                    "preview_image": preview_image,
                    "file_type": file_extension,
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "file_size": len(content)
                }
            ) for chunk in chunks]
            
            # Create embeddings and store in MongoDB
            embedded_chunks = [{
                "text": doc.page_content,
                "embedding": embedding_model.embed_query(doc.page_content),
                "metadata": doc.metadata
            } for doc in documents]
            
            if embedded_chunks:  # Only insert if there are chunks to insert
                MONGODB_COLLECTION.insert_many(embedded_chunks)
            
        return {"message": "Documents uploaded and processed successfully"}
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def search_context(request: MessageRequest):
    """
    Searches for relevant contexts using cosine similarity.
    """
    try:
        docs = vectorstore.similarity_search(request.message, k=5)
        contexts = [doc.page_content for doc in docs]
        concatenated_context = " ".join(contexts)
        print(f"Contexts: {concatenated_context}")
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