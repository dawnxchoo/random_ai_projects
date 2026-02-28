"""
Currency Conversion Agent
=========================
A simple LangChain agent that uses two tools:
  1. A calculator for math operations
  2. A currency converter with live-ish exchange rates

"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

# ---------------------------------------------------------------------------
# Tool 1: Calculator
# ---------------------------------------------------------------------------

@tool

def calculator(expression: str) -> str:
    """
    Evaluate a math expression and return the result.
    Use standard operators: +, -, *, /, ** (power), and parentheses.
    Examples:
      - "1500 * 0.85"
      - "3000 / 149.5"
      - "(250 + 300) * 1.12"
    """
    allowed = set("0123456789.+-*/() ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains invalid characters. Use only numbers and + - * / ** ( )"
    try:
        result = eval(expression)
        return str(round(result, 4))
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool 2: Live Currency Exchange Rate Lookup
# ---------------------------------------------------------------------------
# Uses the free open.er-api.com API (no API key required).
# Supports 150+ currencies with rates updated daily.
# ---------------------------------------------------------------------------

import requests

@tool

def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """
    Get the live exchange rate between two currencies.
    Returns how much 1 unit of from_currency is worth in to_currency.
    Supports 150+ world currencies.

    Args:
        from_currency: The source currency code (e.g. "USD", "EUR")
        to_currency:   The target currency code (e.g. "JPY", "GBP")
    """
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()
    try:
        resp = requests.get(
            f"https://open.er-api.com/v6/latest/{from_currency}",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("result") != "success":
            return f"Error: API returned unexpected result for '{from_currency}'."
        if to_currency not in data["rates"]:
            return f"Error: '{to_currency}' is not a recognized currency code."

        rate = data["rates"][to_currency]
        return f"1 {from_currency} = {rate} {to_currency} (live rate)"
    except requests.exceptions.RequestException as e:
        return f"Error fetching live rate: {e}"


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

def create_currency_agent():
    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        temperature=0,
    )

    tools = [calculator, get_exchange_rate]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful currency conversion assistant. "
            "When the user asks about currency conversions, use the get_exchange_rate "
            "tool to look up rates, then use the calculator tool to compute the final amounts. "
            "Always show your work clearly: state the exchange rate, the calculation, "
            "and the final result rounded to 2 decimal places. "
            "If the user asks a multi-step question, break it down step by step. "
            "Remember previous conversions in the conversation to answer follow-up questions."
        )),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Keeps last 10 messages (5 exchanges) to enable follow-up questions

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,  
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,       # Set to False to hide the reasoning trace
        max_iterations=10,  # Safety net to prevent infinite loops
        handle_parsing_errors=True,
    )


# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Please set your ANTHROPIC_API_KEY in your .env file.")
        print("   Add this line to .env: ANTHROPIC_API_KEY='your-key-here'")
        return

    agent = create_currency_agent()

    print("💱 Currency Conversion Agent")
    print("=" * 40)
    print("Ask me anything about currency conversions!")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        try:
            result = agent.invoke({"input": user_input})
            output = result["output"]

            if isinstance(output, list):
                output = "\n".join(block["text"] for block in output if block.get("text"))

            print(f"\nAgent: {output}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()