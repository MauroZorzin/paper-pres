from openai import OpenAI
client = OpenAI()
resp = client.responses.create(
    model="gpt-5.1",
    reasoning={"effort": "high"},
    input="Example",
)
print(resp.output_text)
``` :contentReference[oaicite:0]{index=0}  

