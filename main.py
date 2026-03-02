from langchain_ollama import ChatOllama



llm=ChatOllama(
    model="Qwen2.5:latest",
    temperature=0.1,
    k=1,

)

h_msg=input("ask me any about AI/ML:")

message=[
    (
        "system",
        "your are helpful assistant for AI/ML QA do your best sample for example"
    ),
    (
        "human",
        h_msg
    )
]

llm_msg=llm.invoke(message)

print("Response:",llm_msg.content)