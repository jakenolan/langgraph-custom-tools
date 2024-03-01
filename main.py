# Necessary imports
import os
import json
import operator
from typing import TypedDict, Annotated, Sequence
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function

# Import notes toolkit
from notes_toolkit import NotesToolkit

# Load env variables
from dotenv import load_dotenv
load_dotenv()

# Init notes toolkit and create tool executor
notes_toolkit = NotesToolkit()
tools = notes_toolkit.get_tools()
tool_executor = ToolExecutor(tools)

# Create model and bind tools as functions
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"], temperature=0, streaming=True)
functions = [convert_to_openai_function(t) for t in tools]
model = model.bind_functions(functions=functions)

# Create class to track agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Function to decide if agent should continue or return to user
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    else:
        return "continue"

# Functional to call llm model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Function to call chosen tool
def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes to cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint (first node called) as agent
workflow.set_entry_point("agent")

# Add a conditional edge to decide if the cycle continues
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# Add a normal edge from tools to agent to always call agent after tools
workflow.add_edge("action", "agent")

# Compile workflow into a LangChain Runnable
app = workflow.compile()

# Create a custom system prompt to direct the agent
system_prompt = PromptTemplate.from_template("""
    You are a helpful assistant.
    Your main jobs are to manage notes for the user and use them to help your resonses.
    The path for your notes is: <{path}>.
    Always check tool and parameter descriptions to confirm you are using the tool correctly.
""")

# Create the inputs with the system message and human message input (can also include message history)
inputs = {"messages": [
    SystemMessage(content=system_prompt.format(path="./notes/")),
    HumanMessage(content="What drink does Jake like?")
]}

# Invoke the runnable app by passing in the inputs and print out the results
messages = app.invoke(inputs)
print(messages["messages"])