import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

# Validate environment variables
ENDPOINT = os.getenv("ENDPOINT")
OPENAI_KEY = os.getenv("OPENAI_KEY")
MODEL = os.getenv("MODEL")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")

if not all([ENDPOINT, OPENAI_KEY, MODEL, SPOONACULAR_API_KEY]):
    raise EnvironmentError(
        "Missing one or more required environment variables (ENDPOINT, OPENAI_KEY, MODEL, SPOONACULAR_API_KEY).")


def system_message(message):
    return {'role': 'system', 'content': message}


def user_message(message):
    return {'role': 'user', 'content': message}


def tool_message(message, tool_call_id):
    return {'role': 'tool', 'content': str(message), 'tool_call_id': tool_call_id}


def get_recipe(query):
    """Fetch a single recipe using Spoonacular API based on the given query."""
    try:
        url = f'https://api.spoonacular.com/recipes/complexSearch?query={query}&number=1&apiKey={SPOONACULAR_API_KEY}'
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        if 'results' in data and data['results']:
            return data['results'][0]['id'], data['results'][0]['title']
        print("No recipes found for the given query.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching recipe: {e}")
    return None


def get_nutritional_details(recipe_id):
    """Fetch nutritional details for a given recipe ID from Spoonacular API."""
    try:
        url = f'https://api.spoonacular.com/recipes/{recipe_id}/nutritionWidget.json?apiKey={SPOONACULAR_API_KEY}'
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        if 'nutrients' in data:
            return data['nutrients'][:5]
        print("Nutritional details not found.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching nutritional details: {e}")
    return None


def function_definitions():
    """Define function schemas for OpenAI tool calls with clear intent differentiation."""
    return [
        {
            'type': 'function',
            'function': {
                'name': 'findRecipe',
                'description': 'Search for a recipe based on the user query. Use this if the user is interested in '
                               'finding a recipe to cook.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'searchQuery': {'type': 'string', 'description': 'Optimized search query for finding a recipe'}
                    },
                    'required': ['searchQuery']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'getNutritionInfo',
                'description': 'Fetch nutritional information for a specific dish. Use this if the user is asking '
                               'about the'
                               'nutritional content of a dish; only the dish name should be used in the query.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'searchQuery': {'type': 'string',
                                        'description': 'Optimized search query for finding the nutritional '
                                                       'information of a recipe'}
                    },
                    'required': ['searchQuery']
                }
            }
        }
    ]


def main():
    try:
        client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=OPENAI_KEY,
            api_version='2024-08-01-preview'
        )

        messages = [
            system_message(
                'You are an assistant that helps with recipes and nutrition. Analyze user queries to determine if '
                'they want:\n'
                '1. To find a recipe (use "findRecipe")\n'
                '2. Nutritional information (use "getNutritionInfo").\n'
                'Convert the query into an optimized search term (less than 3 words) based on this intent. You can '
                'only choose one intent'
            ),
        ]

        user_query = 'Nutrition of Mutton curry'
        messages.append(user_message(user_query))

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=function_definitions()
        )

        messages.append(response.choices[0].message)

        if response.choices[0].finish_reason == 'tool_calls':
            tool_call = response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            print('Function to call:', function_name)

            query = json.loads(tool_call.function.arguments).get('searchQuery')
            if not query:
                print("No query generated.")
                return

            print('Optimized query:', query)

            # Dictionary to map function names to method names
            action_map = {
                'findRecipe': handle_recipe,
                'getNutritionInfo': handle_nutrition
            }

            # Use the dictionary to call the correct method
            if function_name in action_map:
                tool_call_id = tool_call.id
                tools_result = action_map[function_name](query)
                messages.append(tool_message(tools_result, tool_call_id))

                print(messages)
                final_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=function_definitions()
                )
                print(final_response.choices[0].message.content)
            else:
                print("Unknown function called.")

        else:
            print("No tool calls found.")
            print(response.choices[0].finish_reason)
    except Exception as e:
        print(f"An error occurred: {e}")


def handle_recipe(query):
    recipe = get_recipe(query)
    return recipe[1] if recipe else "No recipe found."


def handle_nutrition(query):
    recipe = get_recipe(query)
    if recipe:
        recipe_id = recipe[0]
        nutritional_details = get_nutritional_details(recipe_id)
        return nutritional_details if nutritional_details else "No nutritional details found."
    else:
        print("No recipe found for nutritional details.")


if __name__ == "__main__":
    main()
