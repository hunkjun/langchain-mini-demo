import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import yaml

with open('prompt.yaml', 'r') as file:
    prompts = yaml.safe_load(file)

system_prompt = prompts['system_prompt']
user_prompt = prompts['user_prompt']

# 初始化模型
model = ChatOpenAI(model="gpt-4o", api_key="sk-xxxxx", base_url="http://172.20.90.102:9000/openai/v1/")


# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 定义对话工作流
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# 添加内存以支持连续对话
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 配置对话线程ID
config = {"configurable": {"thread_id": "abc123"}}

# 进行对话
def chat(query):
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    return output["messages"][-1].content

# 交互式对话
print(chat(user_prompt))  # 初始化对话

# 交互聊天
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break
    response = chat(user_input)
    print(f"Bot: {response}")