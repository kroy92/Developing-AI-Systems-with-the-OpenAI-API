import json

from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests


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


def assistant_message(message):
    return {
        'role': 'assistant',
        'content': message
    }


def get_recipe(query, api_key):
    url = f'https://api.spoonacular.com/recipes/complexSearch?query={query}&number=1&apiKey={api_key}'
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        if data['results']:
            return (data['results'][0]['id'], data['results'][0]['title'])
        else:
            return None
    else:
        print(f"Error in fetching recipe: {resp.text}")
        return None


def get_nutritional_details(query, api_key):
    id = get_recipe(query, api_key)[0]
    if not id:
        return None
    else:
        url = f'https://api.spoonacular.com/recipes/{id}/nutritionWidget.json?apiKey={api_key}'
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            data = [data['nutrients'][i] for i in range(0, 5)]
            return data
        else:
            print(f"Error in fetching nutritional details: {resp.text}")
            return None


def function_definitions():
    return [
        {
            'type': 'function',
            'function': {
                'name': 'getSearchQuery',
                'description': 'Retrieves the title of the recipe using the search query',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'searchQuery': {
                            'type': 'string',
                            'description': 'optimised Search query for recipe'
                        }
                    },
                    'required': ['searchQuery']
                }

            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'getNutritionalDetails',
                'description': 'Fetches nutritional details using the search query',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'searchQuery': {
                            'type': 'string',
                            'description': 'optimised Search query for recipe'
                        }
                    },
                    'required': ['searchQuery']
                }
            }
        }

    ]


if __name__ == "__main__":
    load_dotenv()

    endpoint = os.getenv("ENDPOINT")
    key = os.getenv("OPENAI_KEY")
    deployment = os.getenv("MODEL")
    SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")
    # answer = get_recipe('pasta',SPOONACULAR_API_KEY)
    # if not answer:
    #     print("Error in fetching recipe")
    # else:
    #     print(answer)
    # exit(0)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version='2024-08-01-preview'
    )

    msgs = [
        system_message('You are an assistant that helps in finding recipes. You convert a big user query (that might '
                       'have spelling mistakes) for recipes into optimized and correct recipe query in less than 3 '
                       'words'),
    ]

    user_query = [' nutrition of White Sauce Pasta with garlic naan is ?']

    resp = client.chat.completions.create(
        model=deployment,
        messages=msgs + [user_message(user_query[0])],
        tools=function_definitions()
    )

    if resp.choices[0].finish_reason == 'tool_calls':
        function_name = resp.choices[0].message.tool_calls[0].function.name
        print('Function name:', function_name)
        if function_name == 'getSearchQuery':
            query = resp.choices[0].message.tool_calls[0].function.arguments
            query = json.loads(query)
            query = query['searchQuery']
            print('optimized query:', query)
            print(get_recipe(query, SPOONACULAR_API_KEY)[1])
        elif function_name == 'getNutritionalDetails':
            query = resp.choices[0].message.tool_calls[0].function.arguments
            query = json.loads(query)
            query = query['searchQuery']
            print('optimized query:', query)
            print(get_nutritional_details(query, SPOONACULAR_API_KEY))
    else:
        print("No tool calls found")
        print(resp.choices[0].finish_reason)
