from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional
import json
from openai import OpenAI, AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from PIL import Image
import fitz  # PyMuPDF
import io
import base64
from datetime import datetime
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import traceback
from agents import set_default_openai_key, set_tracing_export_api_key, set_tracing_disabled, enable_verbose_stdout_logging, set_default_openai_client

# Updated imports for the Agents SDK
from agents import Agent, RunContextWrapper, Runner, function_tool, ItemHelpers
from agents.run import RunConfig
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from fastapi.responses import StreamingResponse

# Import tools and context from incident_agents.py
from incident_agent import IncidentContext, search_incident_context, query_incidents_db, create_incident_agent

class MessageRequest(BaseModel):
    message: str

# ===========================
# Environment and Configuration
# ===========================
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "ChatMIM"
COLLECTION_NAME = "Incidents"

# Global variables
mongodb_client = None
openai_client = None

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable is not set!")
    print("Tool calls and agent functionality may not work correctly.")

# Set up the API key for both the OpenAI client and tracing
if api_key:
    # This will set the key for both LLM requests and tracing
    set_default_openai_key(api_key, use_for_tracing=True)
    
    # Also explicitly set it for tracing to be sure
    set_tracing_export_api_key(api_key)
    
    # Enable verbose logging for debugging
    enable_verbose_stdout_logging()
    
    print("OpenAI API key configured for both client and tracing")
else:
    # Disable tracing if no API key is available
    set_tracing_disabled(True)
    print("Tracing disabled due to missing API key")

# ===========================
# FastAPI Application Setup
# ===========================
@asynccontextmanager
async def lifespan(app: FastAPI):
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
        mongodb_client.admin.command('ping')
        print("Connected to MongoDB!")
        
        # Initialize the AsyncOpenAI client with the API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set!")
            
        openai_client = AsyncOpenAI(api_key=api_key)
        
        # Set this client as the default for the Agents SDK
        set_default_openai_client(openai_client, use_for_tracing=True)
        
        print("Initialized AsyncOpenAI client and set as default for Agents SDK")
  
        yield
    except Exception as e:
        print(f"Startup error: {e}")
        raise
    finally:
        if mongodb_client:
            mongodb_client.close()
            print("Closed MongoDB connection")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://adorable-macaron-2074b9.netlify.app",
        "https://rag-chat-ui-backend:10000",
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Utility Functions
# ===========================
def create_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 50) -> List[str]:
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

# ===========================
# Chat Endpoint
# ===========================
@app.post("/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        # Create agent context with necessary clients
        agent_context = IncidentContext(
            mongodb_client=mongodb_client, 
            openai_client=openai_client
        )
        
        # Create the incident agent
        incident_agent = create_incident_agent()
        
        # Configure the run with tracing disabled
        run_config = RunConfig(
            workflow_name="Incident Analysis",
            model="gpt-4-1106-preview",
            tracing_disabled=False
        )
        
        async def generate():
            try:
                # Run the agent with streaming
                result = Runner.run_streamed(
                    starting_agent=incident_agent,
                    input=request.message,
                    context=agent_context,
                    max_turns=10,
                    run_config=run_config
                )
                
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        # Stream the raw response events directly
                        yield f"data: {json.dumps({'content': event.data.delta})}\n\n"
                    elif event.type == "run_item_stream_event":
                        if event.item.type == "tool_call_item":
                            # Notify about tool usage
                            yield f"data: {json.dumps({'tool': 'Tool was called'})}\n\n"
                        elif event.item.type == "tool_call_output_item":
                            # Send tool outputs
                            yield f"data: {json.dumps({'tool_output': event.item.output})}\n\n"
                        elif event.item.type == "message_output_item":
                            # Send complete messages using ItemHelpers
                            yield f"data: {json.dumps({'message': ItemHelpers.text_message_output(event.item)})}\n\n"
                    elif event.type == "agent_updated_stream_event":
                        # Notify about agent changes
                        yield f"data: {json.dumps({'agent_update': event.new_agent.name})}\n\n"
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                print(f"Agent execution error: {str(e)}")
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
# Health and Monitoring Endpoints
# ===========================
@app.get("/health")
async def health_check():
    try:
        mongodb_client.admin.command('ping')
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# ===========================
# Incident PDF Metadata Retrival Endpoint
# ===========================
@app.get("/incidents")
async def get_incidents(skip: int = 0, limit: int = 10):
    try:
        collection = mongodb_client[DB_NAME][COLLECTION_NAME]
        
        # First get total count of unique documents
        count_pipeline = [
            {
                "$group": {
                    "_id": "$metadata.filename"
                }
            },
            {
                "$count": "total"
            }
        ]
        
        total_count_result = list(collection.aggregate(count_pipeline))
        total_count = total_count_result[0]['total'] if total_count_result else 0
        
        # Get paginated unique documents
        pipeline = [
            {
                "$group": {
                    "_id": "$metadata.filename",
                    "metadata": {"$first": "$metadata"},
                    "count": {"$sum": 1}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "metadata": 1,
                    "count": 1
                }
            },
            {
                "$skip": skip
            },
            {
                "$limit": limit
            }
        ]
        
        unique_documents = list(collection.aggregate(pipeline))
        
        # Format the response to match what the frontend expects
        formatted_documents = []
        for doc in unique_documents:
            if doc.get('metadata'):
                formatted_documents.append({
                    "metadata": {
                        "filename": doc['metadata'].get('filename'),
                        "preview_image": doc['metadata'].get('preview_image'),
                        "file_type": doc['metadata'].get('file_type'),
                        "upload_timestamp": doc['metadata'].get('upload_timestamp'),
                        "embedding_count": doc.get('count', 0)
                    }
                })
        
        # Return paginated response with metadata
        return {
            "documents": formatted_documents,
            "total": total_count,
            "skip": skip,
            "limit": limit,
            "has_more": (skip + limit) < total_count
        }

    except Exception as e:
        print(f"Error fetching incidents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ===========================
# File Upload Endpoint
# ===========================
@app.post("/upload")
async def upload_documents(files: List[UploadFile]):
    try:
        collection = mongodb_client[DB_NAME][COLLECTION_NAME]
        uploaded_count = 0
        
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
            
            chunks = create_text_chunks(text)
            embedded_chunks = []
            
            for chunk in chunks:
                try:
                    embedding_response = await openai_client.embeddings.create(
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
