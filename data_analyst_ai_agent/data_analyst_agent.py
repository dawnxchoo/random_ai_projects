"""
Data Analyst AI Agent with Conversation Memory
===============================================
An autonomous AI agent that analyzes data by querying PostgreSQL databases
and maintaining conversation context across multiple interactions.

This agent can:
- Execute SQL queries autonomously based on analysis goals
- Remember previous queries and results within a session
- Build on past analysis in follow-up questions
- Provide data-driven insights and recommendations
"""

import os
import json
from pathlib import Path
from decimal import Decimal
from datetime import date, datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from anthropic import Anthropic
from dotenv import load_dotenv


def load_environment():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / '.env'
    return load_dotenv(env_path)


def get_api_key():
    """Retrieve the Anthropic API key from environment variables."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
    return api_key


def get_db_connection_string():
    """Retrieve the PostgreSQL connection string from environment variables."""
    connection_string = os.getenv('POSTGRESQL_NEON_DB')
    if not connection_string:
        raise ValueError("POSTGRESQL_NEON_DB not found in environment variables.")
    return connection_string


def load_system_prompt():
    """Load the data analyst system prompt from file."""
    prompt_path = Path(__file__).parent / 'prompts' / 'data_analyst_ai_agent_prompt.txt'
    with open(prompt_path, 'r') as f:
        return f.read()


def create_db_connection(connection_string):
    """Create a connection to the PostgreSQL database."""
    return psycopg2.connect(connection_string)


def convert_db_types(value):
    """
    Convert database-specific types to JSON-serializable types.
    
    Args:
        value: Any value from database
        
    Returns:
        JSON-serializable version of the value
    """
    if isinstance(value, Decimal):
        return float(value)
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif value is None:
        return None
    else:
        return value


def execute_sql_query(connection, sql_query):
    """
    Execute a SQL query and return results.
    
    Args:
        connection: Database connection
        sql_query (str): SQL query to execute
        
    Returns:
        dict: Contains 'success', 'data' (results), and 'row_count'
    """
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            # Convert to regular dicts and handle database-specific types
            data = []
            for row in results:
                converted_row = {
                    key: convert_db_types(value)
                    for key, value in dict(row).items()
                }
                data.append(converted_row)
            
            # Commit the transaction if successful
            connection.commit()
            
            return {
                "success": True,
                "data": data,
                "row_count": len(data)
            }
    except Exception as e:
        # Rollback the transaction on error so connection is clean for next query
        connection.rollback()
        return {
            "success": False,
            "error": str(e),
            "row_count": 0
        }


def define_tools():
    """
    Define the SQL query tool that the agent can use.
    
    Returns:
        list: Tool definitions in Anthropic's expected format
    """
    return [
        {
            "name": "execute_sql_query",
            "description": "Execute SELECT queries on the social_media_productivity_clean table. Use this to retrieve and analyze data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "The SELECT query to execute on the social_media_productivity_clean table"
                    }
                },
                "required": ["sql_query"]
            }
        }
    ]


def format_tool_result(result):
    """
    Format SQL query results for the AI agent.
    
    Args:
        result (dict): Query result from execute_sql_query
        
    Returns:
        str: Formatted result string
    """
    if not result["success"]:
        return f"Error executing query: {result['error']}"
    
    if result["row_count"] == 0:
        return "Query executed successfully but returned no rows."
    
    # Format as JSON for structured data
    output = f"Query returned {result['row_count']} rows:\n\n"
    output += json.dumps(result["data"], indent=2)
    
    return output


def process_tool_call(tool_name, tool_input, db_connection):
    """
    Execute a tool requested by the AI agent.
    
    Args:
        tool_name (str): Name of the tool to execute
        tool_input (dict): Input parameters for the tool
        db_connection: PostgreSQL database connection
        
    Returns:
        str: Formatted result of the tool execution
    """
    if tool_name == "execute_sql_query":
        sql_query = tool_input["sql_query"]
        print(f"  Executing SQL: {sql_query[:100]}..." if len(sql_query) > 100 else f"  Executing SQL: {sql_query}")
        result = execute_sql_query(db_connection, sql_query)
        return format_tool_result(result)
    else:
        return f"Unknown tool: {tool_name}"


def truncate_messages(messages, max_messages=10):
    """
    Implement window buffer memory to manage conversation length.
    
    Keeps the initial user request and the most recent messages to prevent
    the conversation from exceeding token limits while maintaining context.
    
    Args:
        messages (list): List of message dictionaries (user, assistant, tool_result)
        max_messages (int): Maximum number of recent messages to retain
        
    Returns:
        list: Truncated message list with first message + last N messages
    """
    if len(messages) <= max_messages + 1:
        return messages
    
    # Preserve initial request and keep most recent interactions
    return [messages[0]] + messages[-(max_messages):]


def run_agentic_loop_with_memory(client, system_prompt, messages, tools, db_connection, max_iterations=10):
    """
    Execute the agentic loop with conversation memory.
    
    The agent autonomously decides what queries to run, executes them,
    analyzes results, and builds on previous context within the conversation.
    
    Args:
        client (Anthropic): Anthropic API client
        system_prompt (str): System instructions defining the agent's role
        messages (list): Conversation history (user messages, assistant responses, tool results)
        tools (list): Available tools the agent can use
        db_connection: PostgreSQL database connection
        max_iterations (int): Maximum iterations to prevent infinite loops
        
    Returns:
        tuple: (final_response, updated_messages)
            - final_response (str): Agent's final analysis
            - updated_messages (list): Complete conversation history
    """
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")
        
        # Send conversation history and tools to the AI agent
        print("Calling AI agent...")
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=tools
        )
        
        print(f"Agent response: {response.stop_reason}")
        
        # Interpret the agent's response
        if response.stop_reason == "end_turn":
            print("Agent has completed the analysis")
        elif response.stop_reason == "tool_use":
            print("Agent is querying the database...")
            # Display agent's reasoning if provided
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    print(f"\nAgent reasoning:")
                    print("-" * 60)
                    print(block.text)
                    print("-" * 60)
        
        # Process the agent's decision
        if response.stop_reason == "end_turn":
            # Agent has finished - extract final analysis
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text
            
            # Save final response to conversation history
            messages.append({
                "role": "assistant",
                "content": final_response
            })
            
            return final_response, messages
        
        elif response.stop_reason == "max_tokens":
            # Response exceeded token limit
            print("⚠ Warning: Response hit max_tokens limit")
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text
            
            messages.append({
                "role": "assistant",
                "content": final_response
            })
            
            return final_response + "\n\n[Note: Response may be incomplete due to token limit]", messages
        
        elif response.stop_reason == "tool_use":
            # Agent wants to execute a tool (SQL query)
            
            # Store agent's response in conversation memory
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Execute each tool the agent requested
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
                    
                    print(f"\n  Tool: {tool_name}")
                    
                    # Execute the tool and get results
                    result = process_tool_call(tool_name, tool_input, db_connection)
                    
                    # Log success/failure
                    if "Error executing query" in result:
                        print(f"  ✗ Query failed")
                    else:
                        print(f"  ✓ Query succeeded")
                    
                    print(f"  Result preview: {result[:150]}..." if len(result) > 150 else f"  Result: {result}")
                    
                    # Package result for agent
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result
                    })
            
            # Return tool results to agent for analysis
            messages.append({
                "role": "user",
                "content": tool_results
            })
            
            # Apply memory window to prevent unbounded growth
            messages = truncate_messages(messages, max_messages=10)
        
        else:
            # Unexpected response type
            print(f"Unexpected stop reason: {response.stop_reason}")
            return "Error: Unexpected response from agent", messages
    
    return "Error: Max iterations reached", messages


def main():
    """
    Data Analyst AI Agent - Main execution.
    
    Initializes the agent and runs an autonomous data analysis session
    with conversation memory enabled.
    """
    print("=" * 60)
    print("DATA ANALYST AI AGENT")
    print("=" * 60)
    print()
    
    db_connection = None
    
    try:
        # Initialize agent components
        print("Initializing agent...")
        load_environment()
        api_key = get_api_key()
        connection_string = get_db_connection_string()
        system_prompt = load_system_prompt()
        client = Anthropic(api_key=api_key)
        db_connection = create_db_connection(connection_string)
        tools = define_tools()
        print("✓ Agent initialized")
        print()
        
        print("Starting analysis...")
        print()
        
        # Initialize conversation with analysis request
        messages = [
            {
                "role": "user",
                "content": "Please analyze the social media vs productivity dataset and provide insights."
            }
        ]
        
        # Run the agentic analysis loop
        response, updated_messages = run_agentic_loop_with_memory(
            client=client,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            db_connection=db_connection,
            max_iterations=10
        )
        
        # Display final analysis
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print()
        print(response)
        print()
        print("=" * 60)
        print(f"✓ Session complete ({len(updated_messages)} messages in conversation history)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if db_connection:
            db_connection.close()
            print("\n✓ Database connection closed")


if __name__ == "__main__":
    main()
