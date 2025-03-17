from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional, Literal, Any
import json
from openai import OpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from PIL import Image
import fitz  # PyMuPDF
import io
import base64
from datetime import datetime
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from openai.types.chat import ChatCompletion

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

# Update the tools definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_incidents_db",
            "description": """Query structured incident data from monitoring systems. The data includes:
                - id: Unique incident identifier
                - source: Integration source (e.g., 'Integration Aternity')
                - priority: Incident priority level (1-4)
                - region: Geographic region (e.g., 'ASIA', 'EMEA', 'NA')
                - enhanced_description: Detailed incident description""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "MongoDB query in JSON format. For aggregations, include the full pipeline."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_incident_context",
            "description": "Search through unstructured incident documentation using semantic search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The search query for finding relevant incident documentation"
                    }
                },
                "required": ["message"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assess_and_refine_context",
            "description": "Assess if the current context is sufficient and relevant for the question, then refine it into a concise format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_context": {
                        "type": "string",
                        "description": "Current accumulated context"
                    },
                    "user_question": {
                        "type": "string",
                        "description": "Original user question"
                    }
                },
                "required": ["current_context", "user_question"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

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

async def query_incidents_db(query: str) -> str:
    try:
        collection = mongodb_client[DB_NAME]["Structured_Data"]
        
        # Use GPT to convert natural language to MongoDB query
        query_conversion_messages = [
            {
                "role": "system",
                "content": """Convert natural language queries to MongoDB queries and return the result as a JSON object.
                For aggregations (top N, counting, grouping), ALWAYS use an aggregation pipeline and include proper sorting.
                
                Example formats:
                1. Top N query:
                {"pipeline": [
                    {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 10}
                ]}
                
                2. Simple filter:
                {"query": {"region": "ASIA", "priority": {"$lte": 2}}}
                
                3. Complex aggregation:
                {"pipeline": [
                    {"$match": {"priority": {"$lte": 2}}},
                    {"$group": {"_id": "$source", "count": {"$sum": 1}, "avg_priority": {"$avg": "$priority"}}},
                    {"$sort": {"count": -1}}
                ]}
                
                Common query patterns:
                - "top N": Use $group, $sort, and $limit
                - "maximum/minimum": Use $sort and $limit
                - "count by": Use $group with $sum
                - "average/mean": Use $group with $avg
                
                The data structure is:
                {
                    "id": number,
                    "source": string (technology/system name),
                    "priority": number (1-4),
                    "region": string,
                    "enhanced_description": string
                }
                
                For queries about "top", "most", or "maximum", ALWAYS use an aggregation pipeline with proper sorting and limiting."""
            },
            {
                "role": "user",
                "content": f"Convert this query to MongoDB JSON query: {query}"
            }
        ]

        query_response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=query_conversion_messages,
            response_format={ "type": "json_object" }
        )
        
        # Parse the MongoDB query from GPT's response
        query_object = json.loads(query_response.choices[0].message.content)
        
        # Handle different query types
        if "pipeline" in query_object:
            # Execute aggregation pipeline
            results = list(collection.aggregate(query_object["pipeline"]))
        else:
            # Execute find query
            find_query = query_object.get("query", {})
            results = list(collection.find(find_query, {'_id': 0}))
        
        if not results:
            return "No matching incidents found in structured data."
        
        # Format the results as a string
        formatted_results = []
        for result in results:
            if isinstance(result, dict):
                # Handle aggregation results differently if needed
                if 'count' in result or '_id' in result:
                    # Format aggregation results
                    formatted_results.append(
                        f"Group: {result.get('_id', 'N/A')}\n"
                        f"Count: {result.get('count', 'N/A')}\n"
                        f"Additional Data: {', '.join([f'{k}: {v}' for k, v in result.items() if k not in ['_id', 'count']])}"
                    )
                else:
                    # Format regular document results
                    formatted_results.append(
                        f"ID: {result.get('id')}\n"
                        f"Source: {result.get('source')}\n"
                        f"Priority: {result.get('priority')}\n"
                        f"Region: {result.get('region')}\n"
                        f"Description: {result.get('enhanced_description', '')}...\n"
                    )
        
        print(f"ran query_incidents_db()")
        return "\n".join(formatted_results)
    except json.JSONDecodeError:
        return "Error: Could not parse the query structure"
    except Exception as e:
        print(f"Error querying structured data: {str(e)}")
        return f"Error querying structured data: {str(e)}"

async def search_incident_context(message: str, limit: int = 5) -> str:
    try:
        collection = mongodb_client[DB_NAME][COLLECTION_NAME]
        query_embedding_response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=message
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
        print("ran search_incident_context()")
        return " ".join(contexts)
    except Exception as e:
        return f"Error searching context: {str(e)}"

async def assess_and_refine_context(context: str, question: str) -> str:
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a context refinement assistant. Your job is to:
                1. Remove redundant or duplicate information while preserving unique data points
                2. Format the text to be clearly organized
                3. Keep the original information intact - DO NOT answer the question
                
                For structured data:
                - Keep all unique incident entries
                - Maintain statistical information
                - Format in clear sections
                
                For unstructured data:
                - Remove duplicate passages
                - Keep detailed information intact
                - Maintain specific examples and procedures
                
                If the context is very long (>1000 words):
                - Preserve all technical details (error codes, commands, specific procedures)
                - Keep exact metrics and statistics
                - Summarize descriptive passages while maintaining technical accuracy
                - Combine similar incidents while preserving unique identifiers
                - Keep specific timestamps and incident IDs
                - Maintain critical troubleshooting steps
                
                Example of summarizing while preserving technical details:
                Original: "The authentication service failed at 14:23 GMT with error code E1234. The system attempted three retries using the standard retry policy of 5-second intervals. Each retry resulted in the same error code E1234. Investigation showed the root cause was a misconfigured connection pool size of 15, which was below the required minimum of 25 connections."
                
                Summarized: "Authentication service failure (14:23 GMT) - Error E1234. Three retry attempts at 5s intervals. Root cause: connection pool size 15 (minimum required: 25)"
                
                Return the refined context as a string, maintaining the following format:
                
                Structured Data:
                [Query Purpose]: 
                - Incident 1: [key details]
                - Incident 2: [key details]
                
                Unstructured Documentation:
                [Technical details and procedures]
                [Summarized background information]
                """
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext to refine: {context}"
            }
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            response_format={ "type": "text" }  # Changed to text since we don't need JSON anymore
        )
        
        return response.choices[0].message.content
            
    except Exception as e:
        print(f"Error assessing context: {str(e)}")
        return context  # Return original context if assessment fails

# Define the available tools
AGENT_TOOLS = {
    "search_unstructured_data": {
        "name": "search_unstructured_data",
        "description": "Search through unstructured incident documentation using semantic search. Returns relevant incident descriptions and context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "limit": {"type": "integer", "description": "Number of results to return", "default": 5}
            },
            "required": ["query"]
        }
    },
    "query_structured_data": {
        "name": "query_structured_data",
        "description": "Query structured incident data containing id, source, priority, region, and descriptions. Supports complex MongoDB queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "MongoDB query in natural language"},
                "query_type": {
                    "type": "string", 
                    "enum": ["exact_match", "text_search", "aggregation"],
                    "description": "Type of query to perform"
                }
            },
            "required": ["query", "query_type"]
        }
    }
}

async def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute the specified tool with given input."""
    if tool_name == "search_unstructured_data":
        params = SearchParams(**tool_input)
        return await search_incident_context(params.query, params.limit)
    elif tool_name == "query_structured_data":
        params = StructuredQueryParams(**tool_input)
        return await query_incidents_db(params.query)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

async def run_agent(user_query: str) -> str:
    """Run the ReACT agent to process the user query."""
    messages = [
        {
            "role": "system",
            "content": """You are an incident analysis agent using the ReACT framework. 
            You have access to two data sources:
            1. Unstructured incident documentation with semantic search capabilities
            2. Structured incident data with fields: id, source, priority, region, and description

            Follow this format for your responses:
            Thought: Analyze the task and determine what information is needed
            Action: {tool_name}
            Action Input: {tool_input_json}
            
            After receiving observation:
            Thought: Analyze the observation and determine next steps
            Action/Final Answer: Take another action or provide final answer

            Always explain your reasoning and strategy."""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    max_turns = 5
    turn = 0

    while turn < max_turns:
        response: ChatCompletion = await openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            tools=list(AGENT_TOOLS.values()),
            tool_choice="auto"
        )

        message = response.choices[0].message

        # Check if the agent wants to use a tool
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)

            # Execute the tool
            observation = await execute_tool(tool_name, tool_input)

            # Add the interaction to messages
            messages.extend([
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {"role": "tool", "tool_call_id": tool_call.id, "content": observation}
            ])
        else:
            # Agent has reached a final answer
            return message.content

        turn += 1

    return "Agent reached maximum turns without finding a complete answer."

@app.post("/chat", response_model=Dict[str, str])
async def chat_endpoint(request: MessageRequest):
    try:
        # Run the ReACT agent
        context = await run_agent(request.message)
        
        return {"context": context}
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
