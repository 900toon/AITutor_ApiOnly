from pathlib import Path
from openai import OpenAI
import os
import json
with open("API_Key.json") as f:
    api_key = json.load(f)["API_KEY"]

os.environ['OPENAI_API_KEY'] = api_key

client = OpenAI(api_key=api_key)

response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="切特機批踢 好棒好棒好棒棒 大大好棒棒 大棒棒"
)

filePathTest = "OutputFolder/speech02.wav"
with open(filePathTest, 'wb') as f:
    f.write(response.content)


print("done")