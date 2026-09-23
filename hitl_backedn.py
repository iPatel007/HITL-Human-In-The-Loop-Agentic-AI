from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from tavily import TavilyClient
import os
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
import requests

load_dotenv()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

model = ChatOpenAI(model="gpt-4o-mini")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
# ============================================================
# InMemorySaver and Configuration
# ============================================================
    
config = {"configurable": {"thread_id": "your-unique-thread-id"}}
checkpointer = InMemorySaver()


# ============================================================
# Tools
# ============================================================
    
@tool    
def tavily_search(query: str):
    """ Make web search or we can say tavily search if required for current situation or external information """
    
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(query)
    return response
    
@tool
def calculate_tax(income: float) -> float:
    """Calculate simple interest 15% tax on income."""
    return income * 0.15

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price of any company like (Apple, Google, HAL) 
    using Alpha Vantage with API key in the URL.
    only return stock price only, do not show any other details
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()

def purchase_stock(symbol: str, quantity: int):
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }

    

tools = [tavily_search, calculate_tax, get_stock_price]
model_with_tools = model.bind_tools(tools)


# ============================================================
# chat_node
# ============================================================

def chat_node(state: ChatState):
    user_input = state['messages']
    response = model_with_tools.invoke(user_input)
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tool["name"] for tool in response.tool_calls]
        print(f"👉 Agent is calling tools: {tool_names}")
        
    return {'messages': [response]}    

def human_review_node(state: ChatState):
    # Pauses graph and sends the question to the user
    state = graph.get_state(config)

    if state.next and state.next[0] == "tools":
        # Extract the tool call name from the last message
        last_message = state.values["messages"][-1]
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        
        # HITL confirmation prompt
        user_approval = input(f"🤖 LLM wants to call the tool '{tool_name}'. Proceed? (yes/no): ")
        
        if user_approval.lower() == "yes":
            # Step 3a: Resume execution seamlessly
            print("🚀 Executing tool...")
            for event in graph.stream(None, config):
                print(event)
        else:
            # Step 3b: Reject or intercept (optional)
            print("❌ Tool execution cancelled.")
            # You could inject a refusal message back to the LLM if needed
            
        user_feedback = interrupt(f"Please review: {state['question']}")
        return {"answer": user_feedback}

# ============================================================
# StateGraph
# ============================================================
        
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', ToolNode(tools))

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', "chat_node")

workflow = graph.compile(checkpointer=checkpointer, interrupt_before=['tools'])


# ============================================================
# Runtime Execution Loop with HITL
# ============================================================
while True:
    user_input = input('\nUser - ')
    if user_input.lower() == 'exit':
        break
        
    # 1. Send the user message to the graph
    response = workflow.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)
    
    # 2. Check if the graph paused right before the 'tools' node
    current_state = workflow.get_state(config)
    
    while current_state.next and current_state.next[0] == "tools":
        # Extract details of the requested tool
        last_message = current_state.values["messages"][-1]
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        
        # HITL confirmation prompt in the terminal
        user_approval = input(f"🤖 LLM wants to call the tool '{tool_name}'. Proceed? (yes/no): ")
        
        if user_approval.lower() == "yes":
            print("🚀 Executing tool...")
            # Proceed to tool node naturally
            response = workflow.invoke(None, config=config)
        else:
            print("❌ Tool execution cancelled by user.")
            
            # Create a formal rejection message matching OpenAI specifications
            cancellation_msg = ToolMessage(
                content="Error: Tool execution rejected by the user. Do not attempt to run the tools and provide the answer. Politely tell the user that you cannot proceed without tool approval.",
                tool_call_id=tool_call_id
            )
            
            # Update the graph state with the cancellation message.
            # This satisfies OpenAI's rule by placing the ToolMessage right after the Tool Call.
            workflow.update_state(config, {"messages": [cancellation_msg]}, as_node="tools")
            
            # Resume graph execution. It skips the actual tool execution 
            # and proceeds straight back to the agent node with state in perfect order.
            response = workflow.invoke(None, config=config)
            
        # Refresh state check in case the LLM tries to call another tool right after
        current_state = workflow.get_state(config)
        
    # 3. Print the final answer once the graph reaches END
    print(f"AI - {response['messages'][-1].content}")


