import openai

# 初始化客户端，指向你的本地SGLang服务器
client = openai.OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="none" 
)

# 发送请求
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[
        {"role": "user", "content": "Write a long paragraph about the history of the internet."}
    ],
    temperature=0.7,
    max_tokens=256
)

# usage 信息就在这个 response 对象里
print("--- 完整的响应对象 ---")
print(response)

print("\n--- Usage 详细信息 ---")
print(response.usage)

print("\n--- 模型回复 ---")
print(response.choices[0].message.content)