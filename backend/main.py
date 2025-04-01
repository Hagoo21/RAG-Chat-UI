from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional
import json
from openai import OpenAI, AsyncOpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from PIL import Image
import fitz  # PyMuPDF
import io
import base64
from datetime import datetime
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dataclasses import dataclass
import traceback
from agents import set_default_openai_key, set_tracing_export_api_key, set_tracing_disabled, enable_verbose_stdout_logging, set_default_openai_client

# Updated imports for the Agents SDK
from agents import Agent, RunContextWrapper, Runner, function_tool
from agents.run import RunConfig
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

# ---------------------------
# Pydantic models
# ---------------------------
class MessageRequest(BaseModel):
    message: str

class SQLQueryFunction(BaseModel):
    query: str = Field(..., description="The SQL query to execute")

class ChatFunction(BaseModel):
    message: str = Field(..., description="The message to process using RAG")

class FunctionCall(BaseModel):
    name: str
    arguments: str

class AgentAction(BaseModel):
    tool: str
    tool_input: Dict[str, Any]
    thought: str

class AgentFinish(BaseModel):
    return_value: str
    thought: str

class SearchParams(BaseModel):
    query: str
    limit: Optional[int] = 5

class StructuredQueryParams(BaseModel):
    query: str
    query_type: str

# Structured output model for incident responses
class IncidentResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    query_details: Optional[Dict[str, Any]] = None

# ---------------------------
# Agent Context
# ---------------------------
@dataclass
class IncidentContext:
    mongodb_client: Any
    openai_client: Any
    last_query: str = None
    last_results: Any = None
    search_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.search_history is None:
            self.search_history = []

# ---------------------------
# Environment and Globals
# ---------------------------
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

# ---------------------------
# Utility Functions
# ---------------------------
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

# ---------------------------
# Tool Functions Using function_tool Decorator
# ---------------------------
@function_tool
async def search_incident_context(wrapper: RunContextWrapper[IncidentContext], query: str, limit: int) -> str:
    """Search through unstructured incident documentation using semantic search to find detailed information.
    
    This is the PRIMARY tool for most queries and should be used FIRST for:
    - Finding specific incident details
    - Retrieving troubleshooting steps
    - Getting explanations or context about incidents
    - Understanding procedures, policies, or technical information
    - Answering "how" and "why" questions
    
    The limit parameter controls the number of results (recommended: 5-10).
    """
    try:
        context = wrapper.context
        collection = context.mongodb_client[DB_NAME][COLLECTION_NAME]
        
        # If limit is not provided or invalid, use a reasonable default
        if not limit or limit <= 0:
            limit = 5
        
        query_embedding_response = await context.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = query_embedding_response.data[0].embedding

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_search_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 20,
                    "limit": limit
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        contexts = [doc["text"] for doc in results]
        
        # Store in context for later reference
        context.last_query = query
        context.last_results = contexts
        context.search_history.append({"query": query, "result_count": len(contexts)})
        
        return " ".join(contexts)
        
    except Exception as e:
        print(f"ERROR in search_incident_context: {str(e)}")
        traceback.print_exc()
        return f"Error searching context: {str(e)}"

@function_tool
async def query_incidents_db(wrapper: RunContextWrapper[IncidentContext], query: str, query_type: str) -> str:
    """Query structured incident data to get quantitative information and statistics.
    
    Use this tool ONLY for:
    - Counting incidents by category, region, priority, etc.
    - Finding the "top N" or "most common" incidents
    - Getting numerical metrics or statistics
    - Aggregating data for trends analysis
    - Questions requiring precise counts or measurements
    
    Results are limited to a maximum of 50 documents for performance reasons.
    Valid query_type values: exact_match, text_search, aggregation.
    
    For most other questions, use search_incident_context instead.
    """
    try:        
        context = wrapper.context
        collection = context.mongodb_client[DB_NAME]["Structured_Data"]
        
        # Check if the collection exists and has data
        collection_stats = {}
        try:
            collection_stats = context.mongodb_client[DB_NAME].command("collstats", "Structured_Data")
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
        
        # Create a query conversion agent
        query_conversion_messages = [
            {
                "role": "system",
                "content": (
                    "Convert natural language queries to MongoDB queries. "
                    "Important field definitions:\n"
                    "- 'source': The platform that detected the incident (e.g., 'Datadog', 'Splunk', 'PagerDuty')\n"
                    "- 'enhanced_description': Contains details about the incident including technologies affected\n"
                    "- 'id': Unique identifier for the incident\n"
                    "- 'priority': Incident priority level (lower numbers are higher priority)\n"
                    "- 'region': Geographic region where the incident occurred\n\n"
                    "For queries about specific technologies or applications having issues, use regex search on the 'enhanced_description' field.\n\n"
                    "For 'top N' queries, ALWAYS use an aggregation pipeline with $group, $sort, and $limit stages.\n\n"
                    "Example formats:\n"
                    "1. Top 10 sources with most incidents:\n"
                    '{"pipeline": [{"$group": {"_id": "$source", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}, {"$limit": 10}]}\n'
                    "2. Top applications mentioned in descriptions:\n"
                    '{"pipeline": [{"$match": {"enhanced_description": {"$regex": "application", "$options": "i"}}}, '
                    '{"$group": {"_id": "$source", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}, {"$limit": 10}]}\n'
                    "3. Simple filter:\n"
                    '{"query": {"region": "ASIA", "priority": {"$lte": 2}}}\n'
                    "IMPORTANT: For queries about 'top', 'most', or 'maximum', you MUST use an aggregation pipeline with proper sorting and limiting."
                )
            },
            {
                "role": "user",
                "content": f"Convert this query to MongoDB JSON query: {query}"
            }
        ]
        
        
        try:
            # Use structured output format to get JSON directly
            query_response = await context.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=query_conversion_messages,
                response_format={"type": "json_object"}  # Request JSON output format
            )
            
            response_content = query_response.choices[0].message.content
            
            # Parse the JSON response directly
            query_object = json.loads(response_content)
        
                
        except Exception as e:
            return f"Error converting query to MongoDB format: {str(e)}"
        
        # Add limit if not present to prevent excessive results
        try:
            if "pipeline" in query_object:
                has_limit = any("$limit" in str(stage) for stage in query_object["pipeline"])
                if not has_limit:
                    query_object["pipeline"].append({"$limit": 50})  # Default limit
                
                results = list(collection.aggregate(query_object["pipeline"]))
            else:
                find_query = query_object.get("query", {})                
                results = list(collection.find(find_query, {'_id': 0}).limit(50))  # Default limit
                
        except Exception as e:
            return f"Error executing database query: {str(e)}"
        
        if not results:
            return "No matching incidents found in structured data."
        
        # Format results
        try:
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    if 'count' in result or '_id' in result:
                        formatted_results.append(
                            f"Group: {result.get('_id', 'N/A')}\n"
                            f"Count: {result.get('count', 'N/A')}\n"
                            f"Additional Data: {', '.join([f'{k}: {v}' for k, v in result.items() if k not in ['_id', 'count']])}"
                        )
                    else:
                        formatted_results.append(
                            f"ID: {result.get('id')}\n"
                            f"Source: {result.get('source')}\n"
                            f"Priority: {result.get('priority')}\n"
                            f"Region: {result.get('region')}\n"
                            f"Description: {result.get('enhanced_description', '')}...\n"
                        )
                
        except Exception as e:
            return f"Error formatting query results: {str(e)}"
        
        # Store in context for later reference
        context.last_query = query
        context.last_results = formatted_results
        context.search_history.append({"query": query, "query_type": query_type, "result_count": len(results)})
            
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error querying structured data: {str(e)}"

@function_tool
async def assess_and_refine_context(wrapper: RunContextWrapper[IncidentContext], context_text: str, question: str) -> str:
    """Refine and organize context by removing duplicates and formatting clearly.
    
    Use this tool after retrieving information with search_incident_context or query_incidents_db when:
    - The retrieved information contains duplicates
    - The context needs better organization
    - Multiple pieces of information need to be combined coherently
    - The raw context is difficult to understand
    
    This tool helps prepare the final, well-structured response.
    """
    try:
        context = wrapper.context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a context refinement assistant. Your job is to:\n"
                    "1. Remove redundant or duplicate information while preserving unique data points\n"
                    "2. Format the text to be clearly organized\n"
                    "3. Keep the original information intact - DO NOT answer the question\n\n"
                    "For structured data:\n"
                    "- Keep all unique incident entries\n"
                    "- Maintain statistical information\n"
                    "- Format in clear sections\n\n"
                    "For unstructured data:\n"
                    "- Remove duplicate passages\n"
                    "- Keep detailed information intact\n"
                    "- Maintain specific examples and procedures\n\n"
                    "If the context is very long (>1000 words):\n"
                    "- Preserve all technical details (error codes, commands, specific procedures)\n"
                    "- Keep exact metrics and statistics\n"
                    "- Summarize descriptive passages while maintaining technical accuracy\n"
                    "- Combine similar incidents while preserving unique identifiers\n"
                    "- Keep specific timestamps and incident IDs\n"
                    "- Maintain critical troubleshooting steps\n\n"
                    "Return the refined context as a string with clear sections."
                )
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext to refine: {context_text}"
            }
        ]
        
        # Make sure to await the async call
        response = await context.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages
        )
        
        return response.choices[0].message.content
            
    except Exception as e:
        print(f"Error assessing context: {str(e)}")
        return context_text

# ---------------------------
# Create Incident Analysis Agent
# ---------------------------
def create_incident_agent():
    return Agent[IncidentContext](
        name="incident_analysis_agent",
        instructions=(
            "You are an incident analysis agent specializing in retrieving and analyzing incident information. "
            "You have access to two primary data sources:\n"
            "1. Unstructured incident documentation (accessed via search_incident_context)\n"
            "2. Structured incident data with fields: id, source, priority, region, and description (accessed via query_incidents_db)\n\n"
            
            "IMPORTANT TOOL SELECTION GUIDELINES:\n"
            "- Use search_incident_context as your PRIMARY tool for MOST queries. This tool should be your FIRST choice for:\n"
            "  * Finding detailed information about incidents\n"
            "  * Retrieving troubleshooting steps or procedures\n"
            "  * Understanding the context, causes, or impacts of incidents\n"
            "  * Answering questions about 'how' or 'why' something happened\n"
            "  * Getting explanations or technical details\n\n"
            
            "- Use query_incidents_db ONLY for quantitative questions requiring statistics or counts, such as:\n"
            "  * 'How many incidents occurred in region X?'\n"
            "  * 'What are the top 10 sources of incidents?'\n"
            "  * 'Which priority level has the most incidents?'\n"
            "  * Questions explicitly asking for numerical data or trends\n\n"
            
            "- After retrieving information, use assess_and_refine_context to organize and improve the presentation.\n\n"
            
            "PROCESS FOR ANSWERING QUESTIONS:\n"
            "1. Analyze the question to determine if it requires detailed information (use search_incident_context) "
            "   or quantitative data (use query_incidents_db).\n"
            "2. For most questions, start with search_incident_context unless the question explicitly asks for counts, "
            "   statistics, or 'top N' type information.\n"
            "3. If the initial results don't fully answer the question, consider using the other tool or refining your query.\n"
            "4. Use assess_and_refine_context to organize the information into a clear, coherent response.\n"
            "5. Provide a comprehensive answer that directly addresses the user's question.\n\n"
            
            "Always explain your reasoning and strategy. Be thorough in your analysis but concise in your final response."
        ),
        tools=[search_incident_context, query_incidents_db, assess_and_refine_context]
    )

# ---------------------------
# Chat Endpoint Using Runner
# ---------------------------
@app.post("/chat", response_model=Dict[str, Any])
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
            model="gpt-4-1106-preview",  # Use a model known to work well with function calling
            tracing_disabled=False  # Enable tracing now that we've set the API key properly
        )
        
        try:
            # Run the agent
            result = await Runner.run(
                starting_agent=incident_agent,
                input=request.message,
                context=agent_context,
                max_turns=10,
                run_config=run_config
            )
            
            # Return the final output
            return {"context": result.final_output}
            
        except Exception as e:
            print(f"Agent execution error: {str(e)}")
            return {"context": f"I encountered an issue while processing your request: {str(e)}"}
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Endpoints for Health and Incidents
# ---------------------------
@app.get("/health")
async def health_check():
    try:
        mongodb_client.admin.command('ping')
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
    
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

# ---------------------------
# File Upload Endpoint
# ---------------------------
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
