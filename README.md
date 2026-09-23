# LangGraph Tool-Calling Agent with Human-in-the-Loop

A practical **LangGraph + LangChain + OpenAI** terminal chatbot that demonstrates:

- LLM tool calling
- Web search with Tavily
- Tax calculation with a custom tool
- Stock-price lookup with Alpha Vantage
- Human approval before executing tools
- Tool rejection handling with `ToolMessage`
- LangGraph checkpointing with `InMemorySaver`
- Repeated tool calls within the same conversation thread

This project is primarily a **learning/practice project** for understanding how an agent can decide when to call tools and how a human can approve or reject tool execution.

---

## Architecture

The application uses a simple LangGraph state machine:

```text
                    ┌──────────────────┐
                    │      START       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    chat_node     │
                    │  OpenAI LLM      │
                    └────────┬─────────┘
                             │
                    tools_condition
                       ┌─────┴─────┐
                       │           │
                  Tool call      No tool
                       │           │
                       ▼           ▼
                ┌────────────┐    END
                │   tools    │
                │ ToolNode   │
                └──────┬─────┘
                       │
                       ▼
                  chat_node
```

When the LLM requests a tool, the graph is compiled with:

```python
interrupt_before=["tools"]
```

The runtime then pauses before tool execution and asks the user:

```text
🤖 LLM wants to call the tool 'tavily_search'. Proceed? (yes/no):
```

If the user approves, the tool executes.

If the user rejects the request, a `ToolMessage` is added to the graph state explaining that the tool execution was rejected, and the LLM receives that result.

---

## Features

### 1. OpenAI LLM

The project uses:

```python
ChatOpenAI(model="gpt-4o-mini")
```

The model is configured with the available tools using:

```python
model_with_tools = model.bind_tools(tools)
```

The LLM decides whether it can answer directly or needs one of the registered tools.

---

### 2. Tavily Web Search

The `tavily_search` tool provides external web search capability.

```python
@tool
def tavily_search(query: str):
    ...
```

It is useful for questions requiring current or external information.

The Tavily API key is read from the environment:

```python
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
```

---

### 3. Tax Calculation Tool

A simple custom calculation tool is included:

```python
@tool
def calculate_tax(income: float) -> float:
    return income * 0.15
```

It calculates a fixed 15% tax for demonstration purposes.

> **Important:** This is only a demonstration calculation and is not intended to represent real tax rules or tax advice.

---

### 4. Stock Price Tool

The project includes a stock-price tool using Alpha Vantage:

```python
@tool
def get_stock_price(symbol: str) -> dict:
    ...
```

Example queries:

```text
What is Apple's stock price?
Get the current price of HAL.
```

The tool sends a request to Alpha Vantage and returns the API response.

> **Important:** The current source code contains an Alpha Vantage API key directly in the URL. Before publishing this repository to GitHub, move that key into an environment variable and rotate/revoke the exposed key if it is real.

Recommended approach:

```env
ALPHA_VANTAGE_API_KEY=your_api_key
```

and then:

```python
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
```

---

## Human-in-the-Loop (HITL)

One of the main purposes of this project is demonstrating **human approval before tool execution**.

The graph is compiled as:

```python
workflow = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]
)
```

This means that when the LLM decides to call a tool, LangGraph pauses before the `tools` node.

The application checks the current graph state:

```python
current_state = workflow.get_state(config)
```

Then it extracts the requested tool:

```python
last_message = current_state.values["messages"][-1]
tool_call = last_message.tool_calls[0]
```

The user is asked for approval.

### Approved tool call

If the user enters:

```text
yes
```

the graph resumes:

```python
response = workflow.invoke(None, config=config)
```

The requested tool is executed.

### Rejected tool call

If the user enters anything other than `yes`, the application creates a `ToolMessage`:

```python
cancellation_msg = ToolMessage(
    content="Error: Tool execution rejected by the user...",
    tool_call_id=tool_call_id
)
```

The message is then added to the graph state before the graph resumes.

This preserves the expected relationship between the assistant's tool call and the corresponding tool result/rejection message.

---

## State Management

The application uses a simple state definition:

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The `messages` list stores the conversation history, including:

- Human messages
- AI messages
- Tool calls
- Tool messages

`add_messages` allows LangGraph to append and manage messages correctly.

---

## Checkpointing

The project currently uses:

```python
checkpointer = InMemorySaver()
```

and a fixed thread configuration:

```python
config = {
    "configurable": {
        "thread_id": "your-unique-thread-id"
    }
}
```

The thread ID allows LangGraph to maintain state for the conversation.

### Important limitation

`InMemorySaver` stores checkpoints only in memory.

Therefore:

- Restarting the application loses the conversation state.
- It is suitable for learning and local demonstrations.
- It is not suitable for persistent production conversations.

For production, a persistent checkpointer such as SQLite or another supported database-backed solution can be used.

---

## Project Flow

A typical interaction looks like this:

```text
User
  │
  ▼
chat_node
  │
  ├── Direct answer ───────────────► END
  │
  └── Tool required
          │
          ▼
     Human approval
       │       │
      yes      no
       │       │
       ▼       ▼
   Execute    ToolMessage
     tool      rejection
       │       │
       └───┬───┘
           ▼
       chat_node
           │
           ▼
       Final answer
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the repository URL with your actual GitHub repository.

---

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

---

### 3. Install dependencies

Install the required packages:

```bash
pip install langgraph langchain langchain-openai python-dotenv tavily-python requests
```

Depending on the versions you use, you may also prefer to maintain a `requirements.txt` file:

```bash
pip freeze > requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
```

### Do not commit `.env`

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

The OpenAI key is automatically used by `ChatOpenAI`.

Tavily reads:

```python
os.getenv("TAVILY_API_KEY")
```

The current stock-price implementation should also be changed to read the Alpha Vantage key from the environment instead of keeping it in source code.

---

## Running the Application

Run the Python file containing the agent:

```bash
python main.py
```

You should see:

```text
User -
```

Enter a question.

For example:

```text
User - Calculate tax for an income of 50000
```

The LLM may request the tax tool.

The application will ask:

```text
🤖 LLM wants to call the tool 'calculate_tax'. Proceed? (yes/no):
```

Enter:

```text
yes
```

The tool executes and the LLM generates the final response.

---

## Example Interactions

### Direct LLM response

```text
User - What is LangGraph?

AI - LangGraph is a framework...
```

No tool is required.

---

### Tax calculation

```text
User - Calculate tax on 100000

🤖 LLM wants to call the tool 'calculate_tax'. Proceed? (yes/no): yes

🚀 Executing tool...

AI - The calculated 15% tax is 15000.
```

---

### Web search

```text
User - Search the web for the latest information about LangGraph.

🤖 LLM wants to call the tool 'tavily_search'. Proceed? (yes/no): yes

🚀 Executing tool...

AI - ...
```

---

### Rejecting a tool

```text
User - Calculate tax on 100000

🤖 LLM wants to call the tool 'calculate_tax'. Proceed? (yes/no): no

❌ Tool execution cancelled by user.

AI - I cannot perform that calculation because the requested tool execution was not approved.
```

The exact wording of the final response depends on the LLM.

---

## Exit the Application

Type:

```text
exit
```

The terminal loop will terminate.

---

## Tools Included

| Tool | Purpose | External API |
|---|---|---|
| `tavily_search` | Web search | Tavily |
| `calculate_tax` | Demonstration tax calculation | No |
| `get_stock_price` | Stock quote lookup | Alpha Vantage |

---

## Important Code Review Notes

This repository is a good learning example, but there are several things to improve before treating it as production-ready.

### 1. API key is hard-coded

The current code contains an Alpha Vantage API key directly in the source:

```python
apikey=C9PE94QUEW9VWGFM
```

This should **not** be committed to GitHub.

Use:

```env
ALPHA_VANTAGE_API_KEY=...
```

instead.

If the key is real, rotate it before publishing the repository.

---

### 2. `purchase_stock()` is currently not registered as a tool

The function exists:

```python
def purchase_stock(symbol: str, quantity: int):
    ...
```

but the actual tool list is:

```python
tools = [
    tavily_search,
    calculate_tax,
    get_stock_price
]
```

Therefore `purchase_stock()` is **not available to the LLM**.

If the intention is to demonstrate a purchase operation with human approval, it needs to be deliberately integrated into the tool system after the security and safety behavior has been designed.

---

### 3. `human_review_node()` is not part of the graph

The code defines:

```python
def human_review_node(state: ChatState):
    ...
```

but the node is never added to the graph.

The active HITL implementation is instead handled by:

```python
interrupt_before=["tools"]
```

and the runtime loop.

Therefore `human_review_node()` is currently unused/dead code and could be removed or redesigned as a separate graph node.

---

### 4. `human_review_node()` references state that is not defined

Inside the function:

```python
state['question']
```

is used.

However, `ChatState` contains only:

```python
messages
```

There is no `question` field.

If this unused function is eventually activated, the state definition and node implementation need to be updated.

---

### 5. In-memory checkpointing is temporary

The application uses:

```python
InMemorySaver()
```

This is excellent for experimentation, but state disappears when the process stops.

A persistent checkpointer should be considered for a real application.

---

### 6. Stock API error handling should be improved

The current implementation directly calls:

```python
r = requests.get(url)
return r.json()
```

A production implementation should handle:

- Network errors
- HTTP errors
- API rate limits
- Invalid symbols
- Missing API keys
- API error responses
- Request timeouts

For example, a timeout should be specified:

```python
requests.get(url, timeout=10)
```

---

### 7. Tavily client creation can be improved

The Tavily client is currently created inside the tool on every call:

```python
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
```

For a larger application, the client can be initialized once and reused.

---

### 8. Tool results should be controlled

Some tools return raw API responses.

For production applications, it is generally better for tools to return only the information the agent actually needs.

For example, instead of returning an entire stock API response, the tool could return a structured result containing only:

```text
symbol
price
timestamp
status
```

---

### 9. Tool input validation should be strengthened

Tools such as stock lookup should validate input before making external requests.

Examples:

- Empty symbol
- Invalid symbol format
- Unsupported operation
- Invalid numeric values

---

## Security Considerations

Before publishing this repository:

1. Remove all API keys from source code.
2. Rotate any key that has already been exposed.
3. Add `.env` to `.gitignore`.
4. Add `.env.example` containing placeholder values only.
5. Never commit real credentials.
6. Add timeouts to external HTTP requests.
7. Validate external API responses.
8. Consider rate limiting for public deployments.

Example `.env.example`:

```env
OPENAI_API_KEY=
TAVILY_API_KEY=
ALPHA_VANTAGE_API_KEY=
```

---

## Suggested Future Improvements

This project can be extended into a more complete agent architecture.

### Persistence

Replace:

```python
InMemorySaver()
```

with a persistent checkpointer.

### Streaming

Use LangGraph streaming APIs to display intermediate agent/tool activity.

### Better tool tracing

Display:

```text
Tool requested
Tool approved
Tool executed
Tool result
Final answer
```

in a structured UI.

### Multiple tools

Add tools for:

- Calculator
- Weather
- Database queries
- File search
- RAG
- APIs
- Email
- Calendar
- Custom business systems

### Structured tool results

Use predictable schemas and error handling instead of returning arbitrary API responses.

### Web UI

The terminal interface can later be replaced with:

- Streamlit
- FastAPI + frontend
- React
- Flutter
- Other client applications

### Production HITL

A production human-approval workflow can use a web interface instead of:

```python
input(...)
```

This allows an agent to pause and wait for approval from a human through a UI.

---

## Technologies Used

- Python
- LangGraph
- LangChain
- OpenAI
- Tavily
- Alpha Vantage
- Python dotenv
- Requests

---

## Learning Objectives

This project is useful for learning the following LangGraph concepts:

- `StateGraph`
- Graph state
- `START` and `END`
- Conditional edges
- `tools_condition`
- `ToolNode`
- Tool binding
- Tool calls
- `ToolMessage`
- Checkpointing
- Thread IDs
- Graph interruption
- Human-in-the-loop workflows
- Resuming interrupted graphs
- Rejecting tool calls
- Agent/tool execution loops

---

## Project Status

**Status:** Learning / Practice Project

The code demonstrates the core concepts but should be hardened before production use.

The most important changes before public GitHub publication are:

1. Remove the hard-coded Alpha Vantage API key.
2. Add `.env.example`.
3. Add `.gitignore`.
4. Add proper API error handling.
5. Remove or complete the unused `human_review_node()`.
6. Decide whether `purchase_stock()` should be part of the actual tool set.
7. Consider persistent checkpointing for real applications.

---

## License

Add the license that matches how you want others to use this project.

For example, if you choose MIT:

```text
MIT License
```

You should add a corresponding `LICENSE` file to the repository.

---

## Author

**Amit Jarasaniya**

Senior Mobile / Desktop / AI / ML Developer

Specialized in:

- Flutter
- iOS / Android
- Python
- AI/ML
- Generative AI
- RAG
- Agentic AI
- LangChain
- LangGraph

---

## Disclaimer

This repository is intended for educational and demonstration purposes.

The stock-price and tax tools are examples for demonstrating agent/tool integration and should not be treated as financial, investment, tax, or legal advice.
