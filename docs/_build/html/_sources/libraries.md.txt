# Libraries

## HuggingFace Transformers

Direct model loading and inference in Python without an external server.

**Links:**
- [Documentation](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)
- [GitHub](https://github.com/huggingface/transformers)

### Key classes

| Class | Purpose |
|-------|---------|
| `AutoModelForCausalLM` | Loads a causal language model (next-token prediction) |
| `AutoTokenizer` | Loads the tokenizer for a model |
| `.from_pretrained()` | Downloads and caches model/tokenizer from HuggingFace Hub |
| `.apply_chat_template()` | Formats a list of messages into the model's expected format |
| `.generate()` | Performs autoregressive token generation |

### Environment variables

| Variable | Purpose |
|----------|----------|
| `HF_HOME` | Base directory for HuggingFace files (default `~/.cache/huggingface`) |
| `HF_TOKEN` | HuggingFace API token (required for gated models) |

### Example usage

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Load model and tokenizer
model_name = 'google/gemma-2-2b-it'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a chat prompt
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Hello!'},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Generate response
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```


---

## LangChain

High-level framework for building LLM applications with chains, agents, and RAG pipelines.

**Links:**
- [Documentation](https://python.langchain.com/docs)
- [Ollama integration](https://python.langchain.com/docs/integrations/chat/ollama)
- [Other chat integrations](https://docs.langchain.com/oss/python/integrations/chat)

### Key classes

| Class | Purpose |
|-------|----------|
| `ChatOllama` | LLM client that connects to Ollama |
| `SystemMessage` | Sets the model's behavior/persona |
| `HumanMessage` | A user message |
| `AIMessage` | A model response |

### Example usage

```python
from langchain_ollama import ChatOllama

# Create a client
llm = ChatOllama(
    model='qwen2.5:3b',
    base_url='http://localhost:11434',
    temperature=0.7,
)

# Single request
response = llm.invoke('Tell me a joke')
print(response.content)

# Streaming
for chunk in llm.stream('Count to 10'):
    print(chunk.content, end='', flush=True)
```

### Message types

LangChain uses typed message objects for structured conversations:

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

messages = [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content='Hello!'),
    AIMessage(content='Hi there! How can I help?'),
    HumanMessage(content='What is 2+2?'),
]

response = llm.invoke(messages)
```

---

## Gradio

Build web UIs for LLM chat apps with minimal code.

**Links:**
- [Documentation](https://www.gradio.app/docs)
- [GitHub](https://github.com/gradio-app/gradio)
- [ChatInterface guide](https://www.gradio.app/guides/creating-a-chatbot-fast)

### ChatInterface parameters

| Parameter | Purpose |
|-----------|----------|
| `fn` | Function that takes `(message, history)` and returns a response |
| `type` | `'messages'` = structured history with roles, `'tuples'` = simple pairs |
| `title` | Display title |
| `description` | Optional subtitle |
| `examples` | List of example prompts as buttons |
| `additional_inputs` | Extra UI components (textboxes, sliders, etc.) |

### Example usage

```python
import gradio as gr

def chat(message, history):
    # Your LLM call here
    return f'Echo: {message}'

demo = gr.ChatInterface(
    fn=chat,
    title='My Chatbot',
    type='messages',  # Enable structured history
)

demo.launch()
```

