import json

from openai import AzureOpenAI
from dotenv import load_dotenv
import os

from pydantic import BaseModel


class Answer(BaseModel):
    question: str
    answer: str


class Answers(BaseModel):
    answers: list[Answer]


import requests

# Load environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
key = os.getenv("OPENAI_KEY")
deployment = os.getenv("MODEL")


def system_message(message):
    return {'role': 'system', 'content': message}


def user_message(message):
    return {'role': 'user', 'content': message}


def tool_message(message, tool_call_id):
    return {'role': 'tool', 'content': str(message), 'tool_call_id': tool_call_id}


def get_weather(city):
    return 'The weather in ' + city + ' is 75 degrees Fahrenheit. It is sunny with a light breeze.'


def get_favourite_food(city):
    return 'The favourite food in ' + city + ' is pizza.'


if __name__ == "__main__":
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version='2024-08-01-preview'
    )

    message = [system_message('You are an AI assistant that provides detailed information about cities. Respond '
                              'accurately based on the given context.'),
               user_message('What is the temperature in New York and mumbai? What is the favourite food in Mumbai?')]

    response = client.chat.completions.create(
        model=deployment,
        messages=message,
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'getWeather',
                    'description': 'Get the weather information for a specific city',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'city': {
                                'type': 'string',
                                'description': 'The name of the city to get the weather information for'
                            }
                        },
                        'required': ['city']
                    }

                }

            },
            {
                'type': 'function',
                'function': {
                    'name': 'getFavouriteFood',
                    'description': 'Get the favourite food information for a specific city',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'city': {
                                'type': 'string',
                                'description': 'The name of the city to get the favourite food information for'
                            }
                        },
                        'required': ['city']
                    }

                }
            }
        ]
    )

    if response.choices[0].finish_reason == 'tool_calls':
        message.append(response.choices[0].message)
        for i in range(len(response.choices[0].message.tool_calls)):
            tool_call = response.choices[0].message.tool_calls[i]
            tool_call_id = tool_call.id
            tool_call_name = tool_call.function.name
            tool_call_arguments = json.loads(tool_call.function.arguments)
            city = tool_call_arguments['city']
            print(tool_call_name, city)
            if tool_call_name == 'getWeather':
                message.append(tool_message(get_weather(city), tool_call_id))
            elif tool_call_name == 'getFavouriteFood':
                message.append(tool_message(get_favourite_food(city), tool_call_id))
            else:
                print('Unknown function')

    print(message)

    response = client.beta.chat.completions.parse(
        model=deployment,
        messages=message,
        response_format=Answers

    )

    print (response.choices[0].message.content)



