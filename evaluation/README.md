# Incident Management Chatbot Evaluation System

This evaluation system helps assess and improve the performance of the incident management chatbot agent.

## Overview

The evaluation system consists of:

1. **Test Dataset** - A collection of diverse queries covering different agent capabilities
2. **Agent Wrapper** - An instrumented version of the agent that records tool usage and reasoning
3. **Evaluation Metrics** - Quantitative measures of agent performance 
4. **Testing Framework** - Scripts to run the tests and generate reports

## Setup

Before running the evaluation, make sure you have:

1. Set up the MongoDB database with incident data
2. Configured your OpenAI API key in the .env file
3. Installed the required dependencies:

```bash
pip install aiohttp fastapi uvicorn
```

## Running the Evaluation

### Step 1: Start the Enhanced Agent Server

The enhanced agent provides detailed information about its decision-making process:

```bash
cd evaluation
python agent_wrapper.py
```

This will start a server on port 8001 with the instrumented agent.

### Step 2: Run the Evaluation

Once the enhanced agent server is running, you can run the evaluation:

```bash
python run_evaluation.py --output results
```

Options:
- `--api-url`: URL of the evaluation API (default: http://localhost:8001/evaluation/chat)
- `--test-queries`: Path to test queries file (default: test_queries.json)
- `--output`: Directory to save results (default: results)
- `--max-concurrent`: Max concurrent requests (default: 2)

## Metrics

The evaluation assesses the agent on four key dimensions:

1. **Tool Selection** (35%) - Did the agent choose the appropriate tools?
2. **Tool Utilization** (25%) - Did the agent use the tools effectively?
3. **Content Relevance** (30%) - Does the response contain expected information?
4. **Thought Process** (10%) - Is the agent's reasoning logical and clear?

## Interpreting Results

The evaluation generates a JSON report with:

- Overall pass rate and scores
- Breakdown by category and query type
- Detailed results for each query

A score of 0.7 (70%) or higher is considered passing.

## Customizing the Evaluation

### Adding New Test Cases

Edit `test_queries.json` to add new test cases. Each case should include:

- `id`: Unique identifier
- `category`: Type of query (e.g., search_incident_context, query_incidents_db)
- `complexity`: Difficulty level (basic or complex)
- `query`: The actual query text
- `expected_tools`: List of tools the agent should use
- `expected_content`: Keywords expected in the response

### Modifying Evaluation Criteria

Edit `metrics.py` to adjust:

- Scoring weights
- Passing thresholds
- Tool utilization criteria

## Troubleshooting

- If the evaluation fails to connect to the agent, check that the wrapper server is running
- If MongoDB connection errors occur, verify your MongoDB URI in the .env file
- For OpenAI API errors, check your API key and rate limits 