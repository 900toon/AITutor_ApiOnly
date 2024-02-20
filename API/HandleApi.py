from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
import os
import json
with open("API_Key.json") as f:
    api_key = json.load(f)["API_KEY"]

os.environ['OPENAI_API_KEY'] = api_key


llm = ChatOpenAI(
    openai_api_key = api_key,
    temperature = 0,
    model_name = "gpt-3.5-turbo"
)

temp = ConversationEntityMemory(llm = llm, k=10)

Conversation = ConversationChain(
    llm = llm,
    prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory = temp
)   

#conversation
while True:
    userInput = input("user: ")
    output = Conversation.invoke(userInput)
    print(f"GPT: {output['response']}")