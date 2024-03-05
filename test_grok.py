from groq import Groq

client = Groq(
    api_key="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",
)


stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs",
        }
    ],
    model="mixtral-8x7b-32768",
    stream=True

)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        print(content)