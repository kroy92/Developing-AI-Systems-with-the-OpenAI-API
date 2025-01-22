import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

endpoint = os.getenv("ENDPOINT")
key = os.getenv("OPENAI_KEY")
model = os.getenv("MODEL")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=key,
    api_version='2024-08-01-preview'
)

try:

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'user',
                'content': 'json Mumbai is the capital of Chandigarh?'
            },

        ],
        response_format={
            'type': 'json_object'
        }

    )
    print(response.choices[0].message.content)
except openai.AuthenticationError as e:
    print('Authentication failed: ', e)
except Exception as e:
    print('Error: ', e)


