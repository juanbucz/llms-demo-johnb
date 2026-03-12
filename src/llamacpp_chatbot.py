'''Chat bot demo using a llama.cpp server with the OpenAI-compatible API.

Start llama-server with a MoE model, splitting expert layers to CPU:

$ llama-server \
    -m model.gguf \
    --n-gpu-layers 999 \
    --n-cpu-moe 36 \
    -c 0 -fa \
    --host 0.0.0.0 --port 8502 --api-key "$LLAMA_API_KEY"
'''

import os

from openai import OpenAI

# Configuration
api_key = os.environ.get('LLAMA_API_KEY', 'dummy')
base_url = os.environ.get('LLAMA_BASE_URL', 'http://localhost:8502/v1')
temperature = 0.7

system_prompt = (
    'You are a helpful teaching assistant at an AI/ML boot camp. '
    'Answer questions in simple language with examples when possible.'
)

# Initialize the OpenAI client pointing at the llama.cpp server
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# Get the model name from the server
models = client.models.list()
model = models.data[0].id

# Start conversation history with system prompt
history = [{'role': 'system', 'content': system_prompt}]


def main():
    '''Main conversation loop.'''

    print(f'Connected to llama.cpp server at {base_url}')
    print(f'Model: {model}')
    print('Type "exit" to quit.\n')

    # Loop until user types 'exit'
    while True:

        # Get text input from the user
        user_input = input('User: ')

        # If the user types 'exit', break the loop and end the conversation
        if user_input == 'exit':
            break

        # Add the user's message to the conversation history
        history.append({'role': 'user', 'content': user_input})

        # Send the conversation history to the model and get a response
        response = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=temperature,
        )

        assistant_message = response.choices[0].message.content

        # Add the model's response to the conversation history
        history.append({'role': 'assistant', 'content': assistant_message})

        # Print the model's response
        print(f'\n{model}: {assistant_message}\n')


# Main entry point
if __name__ == '__main__':
    main()
