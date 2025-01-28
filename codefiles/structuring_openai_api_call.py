import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken


def system_message(message):
    return {
        'role': 'system',
        'content': message
    }


def user_message(message):
    return {
        'role': 'user',
        'content': message
    }


# Demonstrates retry logic using the tenacity library
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
def get_response(connection: AzureOpenAI, model, messages):
    resp = connection.chat.completions.create(
        model=model,
        messages=messages,
        # response_format={
        #     'type': 'json_object'
        # }
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    load_dotenv()

    endpoint = os.getenv("ENDPOINT")
    key = os.getenv("OPENAI_KEY")
    deployment = os.getenv("MODEL")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version='2024-08-01-preview'

    )
    msgs = [
        system_message("You are java expert. You are given a series of questions to answer. Provide answer to each "),
        #"question in json format."),
        user_message("What is the difference between a JDK and a JRE?"),
        user_message("What is the difference between an interface and an abstract class?"),
        user_message("What is the difference between a checked and an unchecked exception?"),
    ]
    encoding = tiktoken.encoding_for_model(deployment)
    num_token = [len(encoding.encode(msg['content'])) for msg in msgs]
    print(sum(num_token))

    response = get_response(client, deployment, msgs)
    if response:
        print(response)
    else:
        print("Failed to get a response from the API.")
