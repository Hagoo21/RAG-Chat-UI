import json
from typing import Dict, List, Set, Any, Tuple, Optional

class AgentEvaluator:
    """Evaluator for incident management chatbot agent."""
    
    def __init__(self):
        self.results = []
    
    def evaluate_response(self, 
                          query_id: str,
                          query: str, 
                          expected_tools: List[str],
                          expected_content: List[str],
                          agent_response: Dict[str, Any],
                          tool_calls: List[Dict[str, Any]],
                          thought_process: str = None) -> Dict[str, Any]:
        """
        Evaluate a single agent response against expected outcomes.
        
        Args:
            query_id: Unique identifier for the query
            query: The user's query
            expected_tools: List of tools expected to be used
            expected_content: List of content keywords expected in the response
            agent_response: The final response from the agent
            tool_calls: List of tools used by the agent
            thought_process: The agent's reasoning (if available)
            
        Returns:
            Dict containing evaluation metrics
        """
        # Extract actual tools used
        actual_tools = [call["tool"] for call in tool_calls]
        
        # Tool selection accuracy
        tool_selection_score = self._calculate_tool_selection_score(expected_tools, actual_tools)
        
        # Tool utilization score (were the tools used effectively?)
        tool_utilization_score = self._calculate_tool_utilization_score(tool_calls)
        
        # Content relevance
        content_relevance_score = self._calculate_content_relevance(
            expected_content, 
            agent_response.get("context", "")
        )
        
        # Thought process evaluation
        thought_process_score = self._evaluate_thought_process(thought_process) if thought_process else 0.0
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            tool_selection_score, 
            tool_utilization_score,
            content_relevance_score,
            thought_process_score
        )
        
        # Create result entry
        result = {
            "query_id": query_id,
            "query": query,
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
            "expected_content": expected_content,
            "scores": {
                "tool_selection": tool_selection_score,
                "tool_utilization": tool_utilization_score,
                "content_relevance": content_relevance_score,
                "thought_process": thought_process_score,
                "overall": overall_score
            },
            "pass": overall_score >= 0.7  # Threshold for passing
        }
        
        self.results.append(result)
        return result
    
    def _calculate_tool_selection_score(self, expected_tools: List[str], actual_tools: List[str]) -> float:
        """Calculate how well the agent selected the appropriate tools."""
        if not expected_tools:
            return 1.0
            
        # Set operations for precision and recall
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        
        # Calculate precision (what portion of selected tools were expected)
        precision = len(expected_set.intersection(actual_set)) / len(actual_set) if actual_set else 0.0
        
        # Calculate recall (what portion of expected tools were selected)
        recall = len(expected_set.intersection(actual_set)) / len(expected_set) if expected_set else 0.0
        
        # F1 score (harmonic mean of precision and recall)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_tool_utilization_score(self, tool_calls: List[Dict[str, Any]]) -> float:
        """Evaluate how effectively the tools were used."""
        if not tool_calls:
            return 0.0
            
        scores = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            tool_input = call.get("tool_input", {})
            
            # Score based on tool-specific criteria
            if tool_name == "search_incident_context":
                # Check if query is not empty and limit is reasonable
                query_quality = 1.0 if tool_input.get("query", "").strip() else 0.0
                limit_quality = 0.8 if 3 <= tool_input.get("limit", 0) <= 15 else 0.5
                scores.append((query_quality + limit_quality) / 2)
                
            elif tool_name == "query_incidents_db":
                # Check if query and query_type are valid
                query_quality = 1.0 if tool_input.get("query", "").strip() else 0.0
                query_type_quality = 1.0 if tool_input.get("query_type", "") in ["exact_match", "text_search", "aggregation"] else 0.0
                scores.append((query_quality + query_type_quality) / 2)
                
            elif tool_name == "assess_and_refine_context":
                # Check if context_text and question are not empty
                context_quality = 1.0 if tool_input.get("context_text", "").strip() else 0.0
                question_quality = 1.0 if tool_input.get("question", "").strip() else 0.0
                scores.append((context_quality + question_quality) / 2)
                
            else:
                scores.append(0.5)  # Unknown tool
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_content_relevance(self, expected_content: List[str], actual_content: str) -> float:
        """Calculate relevance of the content based on expected keywords."""
        if not expected_content or not actual_content:
            return 0.0
            
        actual_content_lower = actual_content.lower()
        found_keywords = sum(1 for keyword in expected_content if keyword.lower() in actual_content_lower)
        return found_keywords / len(expected_content)
    
    def _evaluate_thought_process(self, thought_process: str) -> float:
        """Evaluate the agent's reasoning process."""
        if not thought_process:
            return 0.0
            
        # Very simple heuristic - could be improved with more sophisticated analysis
        thought_lines = thought_process.strip().split("\n")
        
        # Look for reasoning patterns
        has_tool_selection = any("tool" in line.lower() for line in thought_lines)
        has_reasoning = any(["because" in line.lower() or "reason" in line.lower() 
                            for line in thought_lines])
        has_strategy = any(["first" in line.lower() or "then" in line.lower() 
                           or "next" in line.lower() for line in thought_lines])
        
        score = 0.0
        if has_tool_selection:
            score += 0.4
        if has_reasoning:
            score += 0.3
        if has_strategy:
            score += 0.3
            
        return score
    
    def _calculate_overall_score(self, 
                                tool_selection: float, 
                                tool_utilization: float,
                                content_relevance: float,
                                thought_process: float) -> float:
        """Calculate overall performance score with weighted components."""
        weights = {
            "tool_selection": 0.35,
            "tool_utilization": 0.25,
            "content_relevance": 0.30,
            "thought_process": 0.10
        }
        
        return (
            tool_selection * weights["tool_selection"] +
            tool_utilization * weights["tool_utilization"] +
            content_relevance * weights["content_relevance"] +
            thought_process * weights["thought_process"]
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        if not self.results:
            return {"error": "No evaluations performed"}
            
        # Calculate aggregate metrics
        total_queries = len(self.results)
        passed_queries = sum(1 for result in self.results if result["pass"])
        
        # Average scores
        avg_scores = {
            "tool_selection": sum(r["scores"]["tool_selection"] for r in self.results) / total_queries,
            "tool_utilization": sum(r["scores"]["tool_utilization"] for r in self.results) / total_queries,
            "content_relevance": sum(r["scores"]["content_relevance"] for r in self.results) / total_queries,
            "thought_process": sum(r["scores"]["thought_process"] for r in self.results) / total_queries,
            "overall": sum(r["scores"]["overall"] for r in self.results) / total_queries
        }
        
        # Group results by category
        category_results = {}
        for result in self.results:
            category = result.get("category", "unknown")
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        # Calculate metrics by category
        category_metrics = {}
        for category, results in category_results.items():
            category_metrics[category] = {
                "count": len(results),
                "pass_rate": sum(1 for r in results if r["pass"]) / len(results),
                "avg_overall_score": sum(r["scores"]["overall"] for r in results) / len(results)
            }
        
        return {
            "summary": {
                "total_queries": total_queries,
                "passed_queries": passed_queries,
                "pass_rate": passed_queries / total_queries if total_queries else 0,
                "average_scores": avg_scores
            },
            "category_metrics": category_metrics,
            "detailed_results": self.results
        }
    
    def export_report(self, filename: str) -> None:
        """Export the evaluation report to a JSON file."""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2) 