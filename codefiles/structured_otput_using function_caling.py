from openai import AzureOpenAI
from dotenv import load_dotenv
import os


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


def function_definitions():
    return [
        {
            'type': 'function',
            'function': {
                'name': 'click',
                'description': 'Clicks the button',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'xpath': {
                            'type': 'string',
                            'description': 'The xpath locator of the button to click'
                        }

                    }

                },
                'required': ['xpath']
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'enter',
                'description': 'Enters the text, number or date in the input field',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'xpath': {
                            'type': 'string',
                            'description': 'The xpath locator of the input field'
                        },
                        'value': {
                            'type': 'string',
                            'description': 'The value to enter in the input field'
                        }
                    }
                },
                'required': ['xpath', 'value']
            }
        }
    ]


def get_response(connection, model, msgs):
    resp = connection.chat.completions.create(
        model=model,
        messages=msgs,
        tools=function_definitions()
    )
    if resp.choices[0].finish_reason == 'tool_calls':
        print('Inside')
        all_steps = [(resp.choices[0].message.tool_calls[i].function.name,
                      resp.choices[0].message.tool_calls[i].function.arguments) for i in
                     range(len(resp.choices[0].message.tool_calls))]
        return all_steps

    else:
        print('Outside')
        print(resp.choices[0])


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
    messages = [
        system_message('''You are tasked with interpreting user test cases to generate automation steps. - Each step 
        in the user's description should be converted into one or more function calls if it describes an action that 
        can be automated. - Ignore steps that do not involve interaction with UI elements or are ambiguous. - Use the 
        'click' function for clicking buttons or links. - Use the 'enter' function for typing into input fields, 
        specifying both the locator and the value to input. - Do not generate any output for steps that do not 
        clearly align with these actions. - Ensure the sequence of actions corresponds to the step numbers provided, 
        even if they are out of numerical order in the text. - If a step includes multiple actions, like entering 
        text and then clicking, break it down into separate function calls.''' ),

        user_message('''
        Step 1 : Enter Username as Admin123
        Step 3 : Click on  My Profile image after login
        Step 2 : Enter Password as M@hiGill92 and Click on Sign in Button
        
        '''),
        assistant_message('''[('enter', '{"xpath": "[XPath for Username field]", "value": "Admin123"}'), ('enter', 
        '{"xpath": "[XPath for Password field]", "value": "M@hiGill92"}'), ('click', '{"xpath": "[XPath for Sign in 
        Button]"}'), ('click', '{"xpath": "[XPath for My Profile Image]"')]'''),

        user_message('''
        1: Click Sales in Site Navigation App
        2: Click on the Add New button
        4: Click om Logout button
        3: Enter the name as 'John Doe' and click on the Save button        
        ''')

    ]
    print(get_response(client, deployment, messages))
