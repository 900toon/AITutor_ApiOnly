from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from openai import OpenAI
from langchain_openai import ChatOpenAI
import os
import json

def TurnSoundFileIntoText(api_key, inputFileName):
    client = OpenAI(api_key=api_key)

    filePath = "InputFolder/" + inputFileName
    audio_file = open(filePath, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file, 
        response_format="text"
    )

    return transcript


#output the sound file right into the output folder
def TurnTextIntoSound(api_key, content):
    client = OpenAI(api_key=api_key)

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input= content
    )

    filePathTest = "OutputFolder/outputSoundFile.wav"
    with open(filePathTest, 'wb') as f:
        f.write(response.content)




#=============================================>
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
    INPUT_FILE_NAME = "speech.mp3"
    userInput = TurnSoundFileIntoText(api_key, INPUT_FILE_NAME)
    output = Conversation.invoke(userInput)
    TurnTextIntoSound(api_key, output['response'])
    print(f"GPT: {output['response']}")


