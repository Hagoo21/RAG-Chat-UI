from agents import Agent, RunContextWrapper, function_tool
from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass
from datetime import datetime
from prompts.db_query_agent_instructions import DB_QUERY_AGENT_INSTRUCTIONS
from prompts.incident_agent_instructions import INCIDENT_AGENT_INSTRUCTIONS

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


# ===========================
# Tools
# ===========================
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
        collection = context.mongodb_client["ChatMIM"]["Incidents"]
        
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
        return f"Error searching context: {str(e)}"

@function_tool
async def query_incidents_db(wrapper: RunContextWrapper[IncidentContext], query: str, query_type: Optional[str] = None) -> str:
    """Query structured incident data to get quantitative information and statistics.
    
    This tool automatically determines the appropriate query type based on the query content:
    - Use aggregation for queries about counts, averages, or top N items
    - Use text search for queries about specific terms or phrases
    - Use exact match for queries about specific field values
    
    The tool supports the following types of queries:
    1. Aggregation queries (counts, averages, top N)
    2. Text search queries (regex patterns)
    3. Exact match queries (specific field values)
    4. Date range queries
    5. Complex queries combining multiple conditions
    
    Results are limited to a maximum of 50 documents for performance reasons.
    """
    try:        
        context = wrapper.context
        collection = context.mongodb_client["ChatMIM"]["Structured_Data"]
        
        # Check if the collection exists and has data
        collection_stats = {}
        try:
            collection_stats = context.mongodb_client["ChatMIM"].command("collstats", "Structured_Data")
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
        
        # Create a query conversion agent with enhanced instructions
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
                    "- 'region': Geographic region where the incident occurred\n"
                    "- 'upload_timestamp': When the incident was recorded\n"
                    "- 'resolution_time': Time taken to resolve the incident\n\n"
                    
                    "Query Type Detection:\n"
                    "1. Aggregation queries (use pipeline):\n"
                    "   - Questions about counts, averages, or top N items\n"
                    "   - Questions about distributions or trends\n"
                    "   - Questions requiring grouping or aggregation\n\n"
                    
                    "2. Text search queries (use regex):\n"
                    "   - Questions about specific terms or phrases\n"
                    "   - Questions about technical components\n"
                    "   - Questions about incident descriptions\n\n"
                    
                    "3. Exact match queries:\n"
                    "   - Questions about specific field values\n"
                    "   - Questions about specific regions or sources\n"
                    "   - Questions about specific priority levels\n\n"
                    
                    "Example formats:\n"
                    "1. Aggregation query (top N):\n"
                    '{"pipeline": [{"$group": {"_id": "$source", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}, {"$limit": 10}]}\n'
                    "2. Text search query:\n"
                    '{"query": {"enhanced_description": {"$regex": "API", "$options": "i"}}}\n'
                    "3. Exact match query:\n"
                    '{"query": {"region": "CANADA", "priority": {"$lte": 2}}}\n'
                    "4. Date range query:\n"
                    '{"query": {"upload_timestamp": {"$gte": new Date(new Date().setMonth(new Date().getMonth() - 1))}}}\n'
                    "5. Complex query:\n"
                    '{"query": {"$and": [{"region": "CANADA"}, {"priority": {"$lte": 2}}, {"enhanced_description": {"$regex": "API", "$options": "i"}}]}}\n\n'
                    
                    "IMPORTANT:\n"
                    "- Always include proper limits for aggregation queries\n"
                    "- Use case-insensitive regex searches when appropriate\n"
                    "- For date queries, use proper MongoDB date format\n"
                    "- For complex queries, use $and/$or operators appropriately\n"
                    "- Always include error handling for missing fields"
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
                model="gpt-4o-2024-11-20",
                messages=query_conversion_messages,
                response_format={"type": "json_object"}  # Request JSON output format
            )
            
            response_content = query_response.choices[0].message.content
            
            # Parse the JSON response directly
            query_object = json.loads(response_content)
        
        except Exception as e:
            return f"Error converting query to MongoDB format: {str(e)}\n\nPlease try rephrasing your query or provide more specific details."
        
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
            return f"Error executing database query: {str(e)}\n\nPlease check if your query parameters are valid and try again."
        
        if not results:
            return "No matching incidents found in structured data. Try broadening your search criteria or using different keywords."
        
        # Enhanced results formatting
        try:
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    if 'count' in result or '_id' in result:
                        # Format aggregation results
                        formatted_results.append(
                            f"Group: {result.get('_id', 'N/A')}\n"
                            f"Count: {result.get('count', 'N/A')}\n"
                            f"Additional Data: {', '.join([f'{k}: {v}' for k, v in result.items() if k not in ['_id', 'count']])}"
                        )
                    else:
                        # Format document results with consistent structure
                        formatted_result = []
                        for field in ['id', 'source', 'priority', 'region', 'enhanced_description']:
                            if field in result:
                                formatted_result.append(f"{field.capitalize()}: {result[field]}")
                        
                        # Add any additional fields
                        additional_fields = [f"{k}: {v}" for k, v in result.items() 
                                          if k not in ['id', 'source', 'priority', 'region', 'enhanced_description']]
                        if additional_fields:
                            formatted_result.append(f"Additional Info: {', '.join(additional_fields)}")
                        
                        formatted_results.append("\n".join(formatted_result))
                
        except Exception as e:
            return f"Error formatting query results: {str(e)}\n\nThe query was successful but there was an issue formatting the results."
        
        # Store in context for later reference
        context.last_query = query
        context.last_results = formatted_results
        context.search_history.append({
            "query": query,
            "query_type": query_type or "auto_detected",
            "result_count": len(results),
            "timestamp": datetime.now().isoformat()
        })
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error querying structured data: {str(e)}\n\nPlease try again with a different query or contact support if the issue persists."


# ===========================
# Agent Initialization Functions
# ===========================
def create_db_query_agent():
    return Agent[IncidentContext](
        name="db_query_agent",
        instructions=DB_QUERY_AGENT_INSTRUCTIONS,
        tools=[query_incidents_db]
    )

def create_incident_agent():
    # Create the database query agent
    db_query_agent = create_db_query_agent()
    
    return Agent[IncidentContext](
        name="incident_analysis_agent",
        instructions=INCIDENT_AGENT_INSTRUCTIONS,
        tools=[
            search_incident_context,
            db_query_agent.as_tool(
                tool_name="query_structured_data",
                tool_description="Query structured incident data for quantitative information and statistics"
            )
        ]
    ) 