import json
import asyncio
import sys
import os
import time
import traceback
from typing import Dict, List, Any, Optional
import argparse
import logging
import aiohttp
from datetime import datetime

# Add parent directory to path so we can import from sibling directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the evaluator
from metrics import AgentEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent_evaluation")

class AgentTester:
    """Test runner for incident management chatbot agent."""
    
    def __init__(self, api_url: str, test_queries_path: str):
        """
        Initialize the tester.
        
        Args:
            api_url: URL of the chat API endpoint
            test_queries_path: Path to the JSON file containing test queries
        """
        self.api_url = api_url
        self.test_queries_path = test_queries_path
        self.evaluator = AgentEvaluator()
        self.test_queries = []
        self.session = None
        
    async def load_test_queries(self) -> None:
        """Load test queries from JSON file."""
        try:
            with open(self.test_queries_path, 'r') as f:
                data = json.load(f)
                self.test_queries = data.get("test_queries", [])
                logger.info(f"Loaded {len(self.test_queries)} test queries")
        except Exception as e:
            logger.error(f"Failed to load test queries: {str(e)}")
            raise
    
    async def initialize(self) -> None:
        """Initialize the tester."""
        await self.load_test_queries()
        self.session = aiohttp.ClientSession()
        logger.info("Initialized agent tester")
        
    async def close(self) -> None:
        """Close any open resources."""
        if self.session:
            await self.session.close()
        logger.info("Closed agent tester")
        
    async def get_agent_response(self, query: str) -> Dict[str, Any]:
        """
        Send a query to the agent and get the response.
        
        Args:
            query: The query to send to the agent
            
        Returns:
            The agent's response
        """
        try:
            async with self.session.post(
                self.api_url,
                json={"message": query, "capture_details": True},
                timeout=120  # Increased timeout for complex queries
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API request failed with status {response.status}: {error_text}")
                    return {"error": f"API request failed: {error_text}"}
                    
                return await response.json()
                
        except Exception as e:
            logger.error(f"Error calling agent API: {str(e)}")
            return {"error": str(e)}
    
    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            test_case: The test case to run
            
        Returns:
            The test result
        """
        query_id = test_case.get("id", "unknown")
        query = test_case.get("query", "")
        expected_tools = test_case.get("expected_tools", [])
        expected_content = test_case.get("expected_content", [])
        
        logger.info(f"Running test {query_id}: {query}")
        
        # Record start time
        start_time = time.time()
        
        # Send query to agent
        agent_response = await self.get_agent_response(query)
        
        # Record elapsed time
        elapsed_time = time.time() - start_time
        
        # Extract tool calls and thought process from the enhanced response
        tool_calls = agent_response.get("tool_calls", [])
        thought_process = agent_response.get("thought_process", "")
        
        # Evaluate the response
        evaluation = self.evaluator.evaluate_response(
            query_id=query_id,
            query=query,
            expected_tools=expected_tools,
            expected_content=expected_content,
            agent_response=agent_response,
            tool_calls=tool_calls,
            thought_process=thought_process
        )
        
        # Add response time to the evaluation
        evaluation["response_time"] = elapsed_time
        
        # Add the category from the test case
        evaluation["category"] = test_case.get("category", "unknown")
        
        logger.info(f"Completed test {query_id} with overall score: {evaluation['scores']['overall']:.2f}")
        
        return evaluation
    
    async def run_all_tests(self, max_concurrent: int = 2) -> List[Dict[str, Any]]:
        """
        Run all test cases.
        
        Args:
            max_concurrent: Maximum number of concurrent tests
            
        Returns:
            List of test results
        """
        logger.info(f"Starting evaluation run with {len(self.test_queries)} queries")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_run_test(test_case):
            async with semaphore:
                return await self.run_single_test(test_case)
        
        # Run tests with bounded concurrency
        tasks = [bounded_run_test(test) for test in self.test_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Test {i} failed with error: {str(result)}")
                logger.error(traceback.format_exception(type(result), result, result.__traceback__))
            else:
                valid_results.append(result)
        
        logger.info(f"Completed {len(valid_results)} tests successfully")
        
        return valid_results
    
    def generate_and_export_report(self, output_path: str) -> None:
        """
        Generate and export the evaluation report.
        
        Args:
            output_path: Path where to save the report
        """
        # Generate a timestamped filename if only a directory is provided
        if os.path.isdir(output_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"agent_evaluation_report_{timestamp}.json")
            
        # Export the report
        self.evaluator.export_report(output_path)
        logger.info(f"Exported evaluation report to {output_path}")
        
        # Also export a summary to the console
        report = self.evaluator.generate_report()
        summary = report["summary"]
        
        logger.info("\n" + "="*50)
        logger.info(f"EVALUATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total queries: {summary['total_queries']}")
        logger.info(f"Passed queries: {summary['passed_queries']} ({summary['pass_rate']*100:.1f}%)")
        logger.info("\nAverage scores:")
        for metric, score in summary["average_scores"].items():
            logger.info(f"  {metric}: {score:.2f}")
        logger.info("="*50)
        
        # Category breakdowns
        logger.info("\nPERFORMANCE BY CATEGORY")
        logger.info("="*50)
        for category, metrics in report.get("category_metrics", {}).items():
            logger.info(f"Category: {category}")
            logger.info(f"  Count: {metrics['count']}")
            logger.info(f"  Pass rate: {metrics['pass_rate']*100:.1f}%")
            logger.info(f"  Average score: {metrics['avg_overall_score']:.2f}")
            logger.info("-"*50)

async def main():
    parser = argparse.ArgumentParser(description='Run evaluation on incident management chatbot agent')
    parser.add_argument('--api-url', type=str, default='http://localhost:8001/evaluation/chat', 
                      help='URL of the evaluation chat API endpoint')
    parser.add_argument('--test-queries', type=str, default='evaluation/test_queries.json',
                      help='Path to the test queries JSON file')
    parser.add_argument('--output', type=str, default='evaluation/results',
                      help='Path to save the evaluation report')
    parser.add_argument('--max-concurrent', type=int, default=2,
                      help='Maximum number of concurrent test requests')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize tester
    tester = AgentTester(args.api_url, args.test_queries)
    
    try:
        await tester.initialize()
        await tester.run_all_tests(max_concurrent=args.max_concurrent)
        tester.generate_and_export_report(args.output)
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main()) 