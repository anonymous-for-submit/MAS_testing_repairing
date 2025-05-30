import openai
from openai import OpenAI
from openai import AzureOpenAI
import time
import os
# from fastchat.model import load_model, get_conversation_template
# from vllm import LLM as vllm
# from vllm import SamplingParams

def call_deepseek_coder(prompt, model='deepseek-coder', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, echo=False, majority_at=None):
    if model == 'deepseek-coder':
        model = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    openai_api_key = "EMPTY"
    openai_api_base = "http://127.0.0.1:8000/v1/"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    # print('in ds coder!')
    # print('prompt')
    # print(prompt)

    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10

    completions = []
    for i in range(3):
        try:
            # print('***'*30)
            # print(prompt)
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            # print(client.api_key)
            # print(client.base_url)
            # print(max_tokens,temperature,top_p,requested_completions)
            response = client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=requested_completions
            )
            while not response:
                response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions
                    )
            completions.extend([choice.message.content for choice in response.choices])
            # print(completions[0])
            # print('*'*30)
            if len(completions) >= num_completions:
                # print('completion')
                # print(completions[0])
                return completions[:num_completions]
        except Exception as e:
            time.sleep(min(i**2, 60))
            print(e)
            print(prompt)
    raise RuntimeError('Failed to call GPT API')

def call_deepseek(prompt, model='deepseek-chat', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, echo=False, majority_at=None):
    key='sk-81408735c17847fbaaf237b3316bd3ee'
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10

    completions = []
    for i in range(3):
        try:
            # print('***'*30)
            # print(prompt)
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            # print(client.api_key)
            # print(client.base_url)
            # print(max_tokens,temperature,top_p,requested_completions)
            response = client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=requested_completions
            )
            while not response:
                response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions
                    )
            completions.extend([choice.message.content for choice in response.choices])
            # print(completions[0])
            # print('*'*30)
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.RateLimitError as e:
            time.sleep(min(i**2, 60))
            print(e)
        except Exception as e:
            time.sleep(min(i**2, 60))
            print(e)
            print(prompt)
    raise RuntimeError('Failed to call GPT API')

# prompt='hello'
# prompt = [
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": prompt},
#     ]
# print(call_deepseek_v3(prompt))

def call_chatgpt(prompt, model='gpt-35-turbo', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, echo=False, majority_at=None):
    # print('$$$'*200)
    # model='gpt-35-turbo'
    # print('in call gpt')
    # print(model)
    # client = OpenAI()
    client = AzureOpenAI(
    azure_endpoint = "https://hkust.azure-api.net", 
    api_key="b8927c969e8147ea8404003613bbddb6",  
    api_version="2024-02-01"
    )

    # client = AzureOpenAI(
    # azure_endpoint = "https://hkust.azure-api.net", 
    # api_key="b8927c969e8147ea8404003613bbddb6",  
    # api_version="2024-02-01"
    # )
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10

    completions = []
    for i in range(3):
        try:
            # print('***'*30)
            # print(prompt)
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            # print(client.api_key)
            # print(client.base_url)
            # print(max_tokens,temperature,top_p,requested_completions)
            response = client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=requested_completions
            )
            while not response:
                response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions
                    )
            completions.extend([choice.message.content for choice in response.choices])
            # print(completions[0])
            # print('*'*30)
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.RateLimitError as e:
            time.sleep(min(i**2, 60))
        except Exception as e:
            time.sleep(min(i**2, 60))
            print(e)
            print(prompt)
    raise RuntimeError('Failed to call GPT API')



# def call_llama():


#     def ask(self, prompt=None, temperature=0, max_tokens=512):
#         prompts = [prompt]
#         return self.generate_batch(prompts, temperature, max_tokens)[0]

#     def generate_batch(self, prompts, temperature=0, max_tokens=512):
#         prompt_inputs = []
#         for prompt in prompts:
#             conv_temp = get_conversation_template(self.model_path)
#             # self.set_system_message(conv_temp)
#             # 类似于client
            
            
#             # conv_temp.append_message(conv_temp.roles[0], prompt)
#             # conv_temp.append_message(conv_temp.roles[1], None)

#             for dic in self.memory_lst:
#                 if dic['role']=='user':
#                     conv_temp.append_message(conv_temp.roles[0], dic['content'])
#                 elif dic['role']=='assistant':
#                     conv_temp.append_message(conv_temp.roles[0], dic['content'])
#                 else:
#                     conv_temp.set_system_message(dic['content'])
#             conv_temp.append_message(conv_temp.roles[1], None)

#             prompt_input = conv_temp.get_prompt()
#             # print('-'*10+'in generate'+'-'*10)
#             # print(conv_temp)
#             # print('-'*20)
#             # print(prompt_input)
#             prompt_inputs.append(prompt_input)

#         sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
#         results = self.model.generate(
#             prompt_inputs, sampling_params, use_tqdm=False)
#         # print('-'*20)
#         outputs = []
#         for result in results:
#             outputs.append(result.outputs[0].text)
#             # print(result.outputs[0].text)
#         # print('end generate')
#         return outputs
