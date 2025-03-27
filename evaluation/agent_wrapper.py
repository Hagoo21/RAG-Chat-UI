import sys
import os
import json
from typing import Dict, List, Any, Optional
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from openai import AsyncOpenAI

# Add parent directory to path so we can import from the backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the agent components - but NOT the clients
from backend.main import Runner, create_incident_agent, IncidentContext, MessageRequest, RunConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent_wrapper")

# Global variables for clients
mongodb_client = None
openai_client = None

# ---------------------------
# Custom models for the wrapper
# ---------------------------
class EnhancedMessageRequest(BaseModel):
    message: str
    capture_details: bool = True

class ToolCall(BaseModel):
    tool: str
    tool_input: Dict[str, Any]
    thought: str
    result: Optional[str] = None

class EnhancedResponse(BaseModel):
    context: str
    tool_calls: List[ToolCall] = []
    thought_process: str = ""

# ---------------------------
# In-memory recording of agent behavior
# ---------------------------
class AgentRecorder:
    def __init__(self):
        self.tool_calls: List[ToolCall] = []
        self.thought_process: List[str] = []
        
    def record_tool_call(self, tool: str, tool_input: Dict[str, Any], thought: str, result: Optional[str] = None):
        self.tool_calls.append(
            ToolCall(
                tool=tool,
                tool_input=tool_input,
                thought=thought,
                result=result
            )
        )
        self.thought_process.append(thought)
        
    def clear(self):
        self.tool_calls = []
        self.thought_process = []

# Global recorder
recorder = AgentRecorder()

# ---------------------------
# Wrapped tool functions to record usage
# ---------------------------
async def wrapped_search_incident_context(wrapper, query: str, limit: int) -> str:
    """Wrapped version of search_incident_context that records usage."""
    # Record the call
    thought = f"I'll search the incident documentation for information about '{query}' with a limit of {limit} results."
    
    # Call the original function
    from backend.main import search_incident_context
    result = await search_incident_context(wrapper, query, limit)
    
    # Record the call with the result
    recorder.record_tool_call(
        tool="search_incident_context",
        tool_input={"query": query, "limit": limit},
        thought=thought,
        result=result[:100] + "..." if len(result) > 100 else result
    )
    
    return result

async def wrapped_query_incidents_db(wrapper, query: str, query_type: str) -> str:
    """Wrapped version of query_incidents_db that records usage."""
    # Record the call
    thought = f"I'll query the structured incidents database for '{query}' using query type '{query_type}'."
    
    # Call the original function
    from backend.main import query_incidents_db
    result = await query_incidents_db(wrapper, query, query_type)
    
    # Record the call with the result
    recorder.record_tool_call(
        tool="query_incidents_db",
        tool_input={"query": query, "query_type": query_type},
        thought=thought,
        result=result[:100] + "..." if len(result) > 100 else result
    )
    
    return result

async def wrapped_assess_and_refine_context(wrapper, context_text: str, question: str) -> str:
    """Wrapped version of assess_and_refine_context that records usage."""
    # Record the call
    thought = f"I'll refine and organize the retrieved context to better answer the question: '{question}'."
    
    # Call the original function
    from backend.main import assess_and_refine_context
    result = await assess_and_refine_context(wrapper, context_text, question)
    
    # Record the call with the result
    recorder.record_tool_call(
        tool="assess_and_refine_context",
        tool_input={"context_text": context_text[:100] + "..." if len(context_text) > 100 else context_text, "question": question},
        thought=thought,
        result=result[:100] + "..." if len(result) > 100 else result
    )
    
    return result

# ---------------------------
# Create a modified version of the incident agent
# ---------------------------
def create_modified_incident_agent():
    """Create a modified version of the incident agent that records tool usage."""
    from backend.main import Agent
    
    # Get the original agent to extract its instructions
    original_agent = create_incident_agent()
    
    return Agent[IncidentContext](
        name="instrumented_incident_analysis_agent",
        instructions=original_agent.instructions,  # Use the exact same instructions
        tools=[
            wrapped_search_incident_context, 
            wrapped_query_incidents_db, 
            wrapped_assess_and_refine_context
        ]
    )

# ---------------------------
# Create the FastAPI app
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mongodb_client, openai_client
    
    try:
        logger.info("Starting agent wrapper server...")
        
        # Check if environment variables are set
        mongodb_uri = os.getenv("MONGODB_URI")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not mongodb_uri:
            logger.error("MONGODB_URI environment variable is not set!")
            raise ValueError("MONGODB_URI environment variable is not set!")
            
        if not openai_key:
            logger.error("OPENAI_API_KEY environment variable is not set!")
            raise ValueError("OPENAI_API_KEY environment variable is not set!")
        
        # Initialize MongoDB client
        logger.info(f"Connecting to MongoDB at {mongodb_uri[:20]}...")
        mongodb_client = MongoClient(
            mongodb_uri,
            server_api=ServerApi('1'),
            maxPoolSize=5,
            minPoolSize=1,
            maxIdleTimeMS=30000,
            retryWrites=True,
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=5000
        )
        
        # Initialize OpenAI client
        logger.info("Initializing OpenAI client...")
        openai_client = AsyncOpenAI(api_key=openai_key)
        
        # Test MongoDB connection
        try:
            mongodb_client.admin.command('ping')
            logger.info("Connected to MongoDB!")
        except Exception as conn_error:
            logger.error(f"MongoDB connection error: {str(conn_error)}")
            raise
        
        yield
    except ImportError as import_err:
        logger.error(f"Import error during startup: {str(import_err)}")
        logger.error(f"Make sure you can import from backend.main and all dependencies are installed")
        raise
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
    finally:
        if mongodb_client:
            mongodb_client.close()
            logger.info("Closed MongoDB connection")
        logger.info("Shutting down agent wrapper")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Enhanced chat endpoint
# ---------------------------
@app.post("/evaluation/chat", response_model=EnhancedResponse)
async def enhanced_chat_endpoint(request: EnhancedMessageRequest):
    """Enhanced chat endpoint for evaluation that returns detailed information."""
    try:
        # Clear the recorder for this request
        recorder.clear()
        
        # Create agent context with our own clients
        agent_context = IncidentContext(
            mongodb_client=mongodb_client, 
            openai_client=openai_client
        )
        
        # Create the modified incident agent
        incident_agent = create_modified_incident_agent()
        
        # Configure the run with tracing
        run_config = RunConfig(
            workflow_name="Incident Analysis Evaluation",
            model="gpt-4-1106-preview",
            tracing_disabled=False
        )
        
        # Run the agent
        result = await Runner.run(
            starting_agent=incident_agent,
            input=request.message,
            context=agent_context,
            max_turns=10,
            run_config=run_config
        )
        
        # Prepare the enhanced response
        return EnhancedResponse(
            context=result.final_output,
            tool_calls=recorder.tool_calls,
            thought_process="\n".join(recorder.thought_process)
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Run the server
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 