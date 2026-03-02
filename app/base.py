from langchain_ollama import ChatOllama

llm=ChatOllama(
    model="Qwen2.5:latest",
    temperature=0.7, #temperature to reduce the hallucination
    k=0.1
)
human_mesg=input("Hi' Am a AI/ML Chatassistant do you have any questions?...   ")
message=[
    ("system",
     "your helpful assistant to solve AI/ML QA"),
     ("human",
      human_mesg)
]

llm_conn=llm.invoke(message)

print("for your Question Response:\n ",llm_conn.content)