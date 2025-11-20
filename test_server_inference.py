from src import llm_utils

prompt = "Explain Genetic Programming"

print(prompt)

res = llm_utils.submit_inference_server(prompt, llm_model='deepseek')

print(res)