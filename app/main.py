from langchain_ollama import ChatOllama
# while using reasoning purpose update below package and steps
from langchain.messages import HumanMessage
from langchain_core.messages import ChatMessage


llm=ChatOllama(
    model="Qwen2.5:latest"
    # temperature=0.7,
    # k=1,

)

h_msg=input("ask me any about AI/ML:")

message=[
    ChatMessage(
        role="control",content="thinking"), #content=thinking mostly used for mathematical operations
    HumanMessage(h_msg)
]

llm_msg=llm.invoke(message)

print("Response:",llm_msg.content)