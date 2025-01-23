from llama_index.llms.ollama import Ollama

llm = Ollama(model="deepseek-r1")
response = llm.complete("Hello, how are you?")
print(response)