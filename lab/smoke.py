import os, json, time
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
c = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
tools=[{"type":"function","function":{"name":"python","description":"Run python code, returns stdout","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}]
t=time.time()
r = c.chat.completions.create(model="openai/gpt-5.6-luna", messages=[{"role":"system","content":"You have a python tool. Variable docs is a list of strings."},{"role":"user","content":"How many docs contain the word 'alpha'? Use the tool."}], tools=tools, extra_body={"usage":{"include":True}})
print(time.time()-t)
m=r.choices[0].message
print(m.content, m.tool_calls and [(tc.function.name, tc.function.arguments) for tc in m.tool_calls])
print(r.usage)
e = c.embeddings.create(model="openai/text-embedding-3-small", input=["hello world","billing service port"])
print(len(e.data), len(e.data[0].embedding), e.usage)
