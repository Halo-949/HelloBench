import json
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import openai
import time
from peft import PeftModel, PeftConfig
from tqdm import *
import numpy as np
from pyramidkv.monkeypatch import replace_llama,replace_mistral
from pyramidkv.temp_cache import temp_cache

GPT_KEY = "Put your OpenAI API key here"


def run_llama31_8b(task_type_list):
    """
    Run Meta-Llama-3.1-8B-Instruct on HelloBench
    :param task_type_list: list of task types
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "llama31_8b", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f]

        for data_dict in data_list:
            instruction = data_dict["chat_prompt"]
            messages = [{"role": "user", "content": instruction}]
            response = pipeline(messages, max_new_tokens=16384, temperature=0.8)[0]["generated_text"][-1]["content"]

            output_dict = {"id": data_dict["id"], "instruction": instruction, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")

def run_llama31_8b_length_constrained_exps():
    """
    Run Meta-Llama-3.1-8B-Instruct on HelloBench-Heuristic_Text_Generation with different length-constaints
    """
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    # )

    model_path = "/root/autodl-tmp/model/Llama-3.1-8B-Instruct"
    attn_implementation = "flash_attention_2"
    max_capacity_prompts = 64
    method = "SnapKV"   # Support FullKV, PyramidKV, SnapKV, H2O, StreamingLLM, CAM, L2Norm, iter-ada-pyraKV, iter-ada-SnapKV

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left"
    )

    replace_llama(method.lower())
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=attn_implementation
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
        

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    
    task_type_list = ["heuristic_text_generation"]

    model.eval()

    token_sum = 0
    num = 0

    if method != "FullKV":
            if method.lower() in ["snapkv","pyramidkv","h2o","cam", "l2norm", "adakv", "headkv","iter-ada-pyrakv","iter-ada-snapkv"]:
                window_sizes = 8
            elif method.lower() in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4  

            kernel_sizes = 7
            pooling = "maxpool"

            layers = len(model.model.layers)
            if method.lower() == "iter-ada-pyrakv":
                temp_cache.ori_capa = max_capacity_prompts
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
                # model.model.layers[i].self_attn.config.merge = args.merge
                # model.model.layers[i].self_attn.config.floor = args.floor

    for task_type in task_type_list:
        # for length in ["2k", "4k", "8k", "16k"]:
        for length in ["2k"]:
            num = 0
            token_sum = 0
            load_path = "data/length_constrained_data/" + task_type + "_" + length + ".jsonl"
            output_path = os.path.join("results", "llama31_8b_" + method,
                                       task_type + "_" + length + "_results.jsonl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(load_path, "r", encoding="utf-8") as f:
                data_list = [json.loads(line) for line in f]
            for data_dict in tqdm(data_list):
                instruction = data_dict["instruction"]
                messages = [{"role": "user", "content": instruction}]
                if length == "16k":
                    max_tokens = 24000
                else:
                    max_tokens = 8192          # 16384

                response = pipeline(messages, max_new_tokens=max_tokens, temperature=0.8)[0]["generated_text"][-1][
                    "content"]
                
                token_count = len(tokenizer.encode(response))
                print("Word Count:", token_count)
                if token_count < 8192:
                    token_sum += token_count
                    num+=1
                    print("avg Word Count:", token_sum/num)

                output_dict = {"id": data_dict["id"], "instruction": instruction, "response": response}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
            print(output_path,": average output length=", token_sum/num)
        


def run_llama31_70b(task_type_list):
    """
    Run Meta-Llama-3.1-70B-Instruct on HelloBench
    :param task_type_list: list of task types
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "llama31_70b", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            messages = [{"role": "user", "content": chat_prompt}]
            response = pipeline(messages, max_new_tokens=16384, temperature=0.8)[0]["generated_text"][-1]["content"]

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_longwriter(task_type_list):
    """
    Run LongWriter on HelloBench
    :param task_type_list: list of task types
    """
    tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-glm4-9b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-glm4-9b", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, device_map="auto")
    model = model.eval()

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "longwriter", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            query = chat_prompt
            response, _ = model.chat(tokenizer, query, history=[], max_new_tokens=16384, temperature=0.8)

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_llama31_70b_length_constrained_exps():
    """
    Run Meta-Llama-3.1-70B-Instruct on HelloBench-Heuristic_Text_Generation with different length-constaints
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    task_type_list = ["heuristic_text_generation"]

    for task_type in task_type_list:
        for length in ["2k", "4k", "8k", "16k"]:
            load_path = "HelloBench/length_constrained_experiments_data/" + task_type + "_" + length + ".jsonl"
            output_path = os.path.join("results", "llama31_70b",
                                       task_type + "_" + length + "_results.jsonl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(load_path, "r", encoding="utf-8") as f:
                data_list = [json.loads(line) for line in f]
            for data_dict in data_list:
                instruction = data_dict["chat_prompt"]
                messages = [{"role": "user", "content": instruction}]
                if length == "16k":
                    max_tokens = 24000
                else:
                    max_tokens = 16384

                response = pipeline(messages, max_new_tokens=max_tokens, temperature=0.8)[0]["generated_text"][-1][
                    "content"]

                output_dict = {"id": data_dict["id"], "instruction": instruction, "response": response}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_longwriter_length_constrained_exps():
    """
    Run LongWriter on HelloBench-Heuristic_Text_Generation with different length-constaints
    """
    tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-glm4-9b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-glm4-9b", torch_dtype=torch.bfloat16,
                                                trust_remote_code=True, device_map="auto")
    model = model.eval()

    task_type_list = ["heuristic_text_generation"]
    for task_type in task_type_list:
        length_list = ["2k", "4k", "8k", "16k"]
        for length in length_list:
            load_path = "HelloBench/length_constrained_experiments_data/" + task_type + "_" + length + ".jsonl"
            output_path = os.path.join("results", "longwriter", task_type + "_" + length + "_results.jsonl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(load_path, "r", encoding="utf-8") as f:
                data_list = [json.loads(line) for line in f.readlines()]

            for data_dict in data_list:
                chat_prompt = data_dict["chat_prompt"]
                query = chat_prompt
                if length == "16k":
                    max_tokens = 24000
                else:
                    max_tokens = 16384
                response, _ = model.chat(tokenizer, query, history=[], max_new_tokens=max_tokens, temperature=0.8)

                output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # task_type_list = ["heuristic_text_generation", "summarization", "text_completion", "chat", "open_ended_qa"]
    task_type_list = ["heuristic_text_generation"]
    # Uncomment this line if you want to use the corresponding model
    run_llama31_8b_length_constrained_exps()
    # run_llama31_70b(task_type_list=task_type_list, model_type="chat")
    # run_longwriter(task_type_list=task_type_list)
    # run_llama31_70b_length_constrained_exps()
    # run_longwriter_length_constrained_exps()