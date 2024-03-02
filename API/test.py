from openai import OpenAI
import os
import json
import time
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_openai import ChatOpenAI

#Get setup file before running:

initFilePath = "D:\\AITutor_onUnity\\AITutor\\AlTutor\\Assets\\DataTransfer\\Initialization.txt"
while(True):
    try:
        with open(initFilePath, "r") as f:
            print(f.read())
            break
    except:
        pass

#strat transfering
print("Initializtion success")
with open("API_Key.json") as f:
    api_key = json.load(f)["API_KEY"]

os.environ['OPENAI_API_KEY'] = api_key

client = OpenAI(api_key=api_key)

audioRecordPath =  "D:\\AITutor_onUnity\\AITutor\\AlTutor\\Assets\\DataTransfer\\PlayerInput"
fileList = os.listdir(audioRecordPath)


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

txtfileNumber = 0
while(True):
    time.sleep(3)
    fileList = os.listdir(audioRecordPath)
    if (len(fileList) == 0): 
        print("nothing in file list............")
        continue
    else:
        transcriptPath = "D:\\AITutor_onUnity\\AITutor\\AlTutor\\Assets\\DataTransfer\\ServerOutput_Text"
        with open(audioRecordPath + f"\\{fileList[0]}", "rb") as f:
            print(f"wavfile fetch success: {fileList[0]}")

            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file= f, 
                response_format="text"
            )
        print("transcript fetch success")
        output = Conversation.invoke(transcript)
        
        with open(transcriptPath + f"\\output{txtfileNumber}.txt", "w+", encoding='utf-8') as f2:
            f2.write(output['response'])

        txtfileNumber+=1
        
        os.remove(audioRecordPath + f"\\{fileList[0]}")
        print(f"chatgpt:  {output['response']}")
        
