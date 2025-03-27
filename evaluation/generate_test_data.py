import os
import json
import argparse
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_data_generator")

# Load environment variables
load_dotenv()

# Categories of test queries
QUERY_CATEGORIES = [
    "search_incident_context",  # Primarily requires search_incident_context
    "query_incidents_db",       # Primarily requires query_incidents_db
    "assess_and_refine_context", # Requires assess_and_refine_context
    "multi_tool",               # Requires multiple tools
    "edge_case",                # Edge cases and unusual requests
    "ambiguous"                 # Queries that could be handled multiple ways
]

# Tools
TOOLS = [
    "search_incident_context",
    "query_incidents_db",
    "assess_and_refine_context"
]

class TestDataGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the test data generator."""
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
                
        self.client = AsyncOpenAI(api_key=api_key)
        
    async def generate_test_queries(self, 
                             num_queries: int = 3, 
                             categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate test queries for the incident management chatbot.
        
        Args:
            num_queries: Number of queries to generate per category
            categories: List of categories to generate queries for. If None, generate for all.
            
        Returns:
            List of generated test queries
        """
        if not categories:
            categories = QUERY_CATEGORIES
            
        all_queries = []
        for category in categories:
            logger.info(f"Generating {num_queries} queries for category: {category}")
            category_queries = await self._generate_category_queries(category, num_queries)
            all_queries.extend(category_queries)
            
        return all_queries
    
    async def _generate_category_queries(self, category: str, num_queries: int) -> List[Dict[str, Any]]:
        """Generate queries for a specific category."""
        prompt = self._create_generation_prompt(category)
        
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Generate {num_queries} test queries for category: {category}"}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        try:
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Ensure the response has the expected format
            if "queries" not in data:
                logger.warning(f"Response for category {category} doesn't contain 'queries' key")
                return []
                
            # Process and validate each query
            processed_queries = []
            for query_data in data["queries"]:
                if not self._validate_query(query_data):
                    continue
                    
                # Add ID and category
                query_data["id"] = f"{category}_{str(uuid.uuid4())[:8]}"
                query_data["category"] = category
                
                processed_queries.append(query_data)
                
            return processed_queries
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for category {category}")
            return []
        except Exception as e:
            logger.error(f"Error generating queries for category {category}: {str(e)}")
            return []
    
    def _validate_query(self, query_data: Dict[str, Any]) -> bool:
        """Validate a query has all required fields."""
        required_fields = ["query", "expected_tools", "expected_content", "complexity"]
        
        for field in required_fields:
            if field not in query_data:
                logger.warning(f"Query missing required field: {field}")
                return False
                
        # Validate expected_tools contains only valid tools
        for tool in query_data.get("expected_tools", []):
            if tool not in TOOLS:
                logger.warning(f"Query contains invalid tool: {tool}")
                return False
                
        return True
    
    def _create_generation_prompt(self, category: str) -> str:
        """Create a prompt for generating queries in a specific category."""
        base_prompt = (
            "You are a test data generator for an incident management chatbot that has access to the following tools:\n"
            "1. search_incident_context - Searches unstructured incident documentation using semantic search\n"
            "2. query_incidents_db - Queries structured incident data (with fields like id, source, priority, region)\n"
            "3. assess_and_refine_context - Refines and organizes context by removing duplicates and formatting\n\n"
            
            "Generate realistic user queries that would test the chatbot's ability to select and use the appropriate tools.\n"
            "For each query, include:\n"
            "- A realistic user question about incident management\n"
            "- Which tools should be used to answer it\n"
            "- What content would be expected in a good response\n"
            "- The complexity level (basic or complex)\n\n"
            
            "Output JSON in this format:\n"
            "{\n"
            '  "queries": [\n'
            "    {\n"
            '      "query": "What are the most common causes of database failures?",\n'
            '      "expected_tools": ["search_incident_context"],\n'
            '      "expected_content": ["database", "failures", "causes"],\n'
            '      "complexity": "basic"\n'
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            
            "Ensure each query feels natural and realistic."
        )
        
        # Add category-specific instructions
        if category == "search_incident_context":
            return base_prompt + (
                "\nFocus on queries that would primarily require searching through unstructured documentation. "
                "These would be questions about specific incidents, procedures, troubleshooting steps, "
                "or explanations of technical concepts. Do not include questions asking for counts, statistics, "
                "or aggregated data."
            )
            
        elif category == "query_incidents_db":
            return base_prompt + (
                "\nFocus on queries that would primarily require querying structured incident data. "
                "These would be questions about statistics, counts, trends, or aggregated information "
                "such as 'How many incidents...', 'What is the most common...', 'Which region has the highest...', etc."
            )
            
        elif category == "assess_and_refine_context":
            return base_prompt + (
                "\nFocus on queries that would require organizing and summarizing information. "
                "These would typically be requests for summaries, comparisons, or synthesizing information "
                "from multiple sources. They often include words like 'summarize', 'compare', or 'explain'."
            )
            
        elif category == "multi_tool":
            return base_prompt + (
                "\nFocus on complex queries that would require using multiple tools together. "
                "These would be questions that need both statistical information AND detailed explanations, "
                "or questions that require getting information and then refining it."
            )
            
        elif category == "edge_case":
            return base_prompt + (
                "\nFocus on unusual or edge case queries that might challenge the chatbot. "
                "These could be questions that are very specific, contain unusual requirements, "
                "or require creative interpretation. They should still be relevant to incident management."
            )
            
        elif category == "ambiguous":
            return base_prompt + (
                "\nFocus on ambiguous queries that could be interpreted in multiple ways. "
                "These would be questions where it's not immediately clear which tool should be used, "
                "or where multiple approaches could be valid."
            )
            
        return base_prompt
    
    async def save_queries(self, queries: List[Dict[str, Any]], output_path: str):
        """Save generated queries to a JSON file."""
        try:
            # If the file exists, append to it
            existing_queries = []
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    data = json.load(f)
                    existing_queries = data.get("test_queries", [])
                    
            # Combine existing and new queries
            all_queries = existing_queries + queries
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump({"test_queries": all_queries}, f, indent=2)
                
            logger.info(f"Saved {len(queries)} new queries to {output_path}")
            logger.info(f"Total queries in file: {len(all_queries)}")
            
        except Exception as e:
            logger.error(f"Error saving queries to {output_path}: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description='Generate test queries for incident management chatbot')
    parser.add_argument('--output', type=str, default='test_queries.json',
                      help='Path to save the generated queries')
    parser.add_argument('--num-queries', type=int, default=2,
                      help='Number of queries to generate per category')
    parser.add_argument('--categories', type=str, nargs='+',
                      help='Categories to generate queries for')
    parser.add_argument('--api-key', type=str,
                      help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Create generator
    generator = TestDataGenerator(api_key=args.api_key)
    
    # Generate queries
    queries = await generator.generate_test_queries(
        num_queries=args.num_queries,
        categories=args.categories
    )
    
    # Save queries
    await generator.save_queries(queries, args.output)

if __name__ == "__main__":
    asyncio.run(main()) 