import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from matheval import evaluator_map, set_client, AIMEEvaluator
import asyncio
import matheval
import humanevaleval
import mbppeval
from huggingface_hub import HfApi

MATH_DATASETS = ["math500","aime2024","gpqa_diamond","gsm8k"]
CODE_DATASETS = ["humaneval","mbpp","livecodebench"]

async def main():
    parser = argparse.ArgumentParser(description='Process some parameters for text generation.')
    parser.add_argument('--dataset', type=str, choices=["math500", "aime2024", "gpqa_diamond", "gsm8k", "humaneval", "mbpp", "livecodebench"], help='Name of dataset')
    parser.add_argument('--model_name', type=str, required=True, default="DeepSeek-R1-Distill-Qwen-1.5B", help='Model name or path')
    parser.add_argument('--max_generated_tokens', type=int, default=10000, help='Limit the number of generated tokens')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--num_samples', type=int, default=1, help='Sampling number')
    parser.add_argument('--num_gpus', type=int, default=1, help='GPU number')
    parser.add_argument('--max_running_requests', type=int, default=None, help='Max number of requests runned together.')
    parser.add_argument('--mem_fraction_static', type=float, default=0.8, help='Max memory to use per gpu.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling probability')
    parser.add_argument('--top_k', type=int, default=30, help='Top-k sampling probability')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing samples')
    parser.add_argument('--end_idx', type=int, default=500, help='End index for processing samples')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save results')
    parser.add_argument('--reeval', action='store_true', help='Enable re-evaluation')
    parser.add_argument('--api_base', type=str, default=None, help='')
    parser.add_argument('--deployment_name', type=str, default=None, help='')
    parser.add_argument('--api_version', type=str, default=None, help='')
    parser.add_argument('--api_key', type=str, default=None, help='')

    parser.add_argument('--push_results_to_hf', action='store_true', help='Enable push to huggingface')
    parser.add_argument('--hf_token', type=str, default=None, help='')
    parser.add_argument('--hf_repo_id', type=str, default=None, help='')

    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model_name
    max_generated_tokens = args.max_generated_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    num_samples = args.num_samples
    num_gpus = args.num_gpus
    max_running_requests = args.max_running_requests
    mem_fraction_static = args.mem_fraction_static
    start_idx = args.start_idx
    end_idx = args.end_idx
    reeval = args.reeval


    print(f"Arguments: {args}")
    
    matheval.set_client(args.api_base, args.deployment_name, args.api_version, args.api_key)

    if dataset == "math500":
        with open("./datasets/math500.json") as f:
            samples = json.load(f)
    elif dataset == "aime2024":
        with open("./datasets/aime2024.json") as f:
            samples = json.load(f)
    elif dataset == "gpqa_diamond":
        with open("./datasets/gpqa_diamond.json") as f:
            samples = json.load(f)
    elif dataset == "gsm8k":
        with open("./datasets/gsm8k.json") as f:
            samples = json.load(f)
    elif dataset == "humaneval":
        with open("./datasets/humaneval.json") as f:
            samples = json.load(f)
    elif dataset == "mbpp":
        with open("./datasets/mbpp.json") as f:
            samples = json.load(f)
    elif dataset == "livecodebench":
        with open("./datasets/livecodebench.json") as f:
            samples = json.load(f)

    else:
        raise ValueError("Invalid dataset name")

    MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

{Question}
""".strip()

    GPQA_QUERY_TEMPLATE = """
Please solve the following multiple-choice question. Please show your choice in the answer field with only the choice letter, e.g.,"answer": "C".

{Question}
""".strip()
    
    CODE_QUERY_TEMPLATE = """
Please solve the programming task below in Python. Code should be wrapped in a markdown code block.

```python
{Question}
```
""".strip()

    MBPP_QUERY_TEMPLATE = """
Please solve the programming task with test cases below in Python. Make sure your code satisfies the following requirements:
1. The function name and signature must match exactly as specified in the test cases.
2. Your code should be wrapped in a markdown code block without including any test cases.

Task:
{Question}

Test Cases:
```python
{TestCases}
```
""".strip()
    def get_lcb_prompt(question_content, starter_code):
        prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
        prompt += f"Question: {question_content}\n\n"
        if starter_code:
            prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            prompt += f"```python\n{starter_code}\n```\n\n"
        else:
            prompt += f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
        return prompt


    
    if not (dataset in CODE_DATASETS and reeval):
        llm = sgl.Engine(model_path=model_name, tp_size=num_gpus, log_level="info", trust_remote_code=True, random_seed=0, max_running_requests=max_running_requests, mem_fraction_static=mem_fraction_static)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_new_tokens": max_generated_tokens, "n": 1} # repeat in input instead of sampling

    os.makedirs(f"{args.output_dir}/results", exist_ok=True)
    results_file = f"{args.output_dir}/results/{model_name.split('/')[-1]}_{dataset}_False_{num_samples}_{temperature}_{top_p}_{top_k}.json"
    results_statistics_file = f"{args.output_dir}/results/{model_name.split('/')[-1]}_{dataset}_False_{num_samples}_{temperature}_{top_p}_{top_k}_statistics.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        print("No results (1)")
        results = []
    done = {entry["idx"] for entry in results}

    print("begin")
    for i in range(start_idx, min(end_idx,len(samples))):
        sample = samples[i]
        if i not in done:
            if dataset in ["aime2024", "math500", "gsm8k"]:
                chat = [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
            elif dataset == "gpqa_diamond":
                chat = [{"role": "user", "content": GPQA_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
            elif dataset == "humaneval":
                chat = [{"role": "user", "content": CODE_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
            elif dataset == "mbpp":
                chat = [{"role": "user", "content": MBPP_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"], TestCases="\n".join(sample["final_answer"]["test_list"]))}]
            elif dataset == "livecodebench":
                chat = [{"role": "user", "content": get_lcb_prompt(question_content=sample["prompt"][0]["value"], starter_code=sample["final_answer"]["starter_code"])}]
            else:
                raise ValueError("Invalid dataset name")

            prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            prompt_list = [prompt] * num_samples
            # outputs = llm.generate([prompt], sampling_params=sampling_params)
            outputs =  await llm.async_generate(prompt_list, sampling_params)
            decoded_text_list = [o["text"] for o in outputs]
            finish_generation_list = [o["meta_info"]["finish_reason"] != "length" for o in outputs]
            
        elif reeval:
            for result in results:
                if result["idx"] == i:
                    decoded_text_list = result["completion"]
                    finish_generation_list = result["finish_generation"]
                    break
            else:
                raise ValueError("No result found for idx: {}".format(i))
        else:
            continue
            
        if dataset in MATH_DATASETS:
            # rule_judge_results = []
            # llm_judge_results = []
            # finally_judge_results = []
            judge_info = []
            passat1_list = []

            for n, decoded_text in enumerate(decoded_text_list):
                rule_judge_result = None
                rule_judge_result, extracted_answer = matheval.evaluator_map[dataset].rule_judge(decoded_text,sample["final_answer"], finish_generation_list[n])
                llm_judge_result = None
                if not rule_judge_result:
                    llm_judge_result = matheval.evaluator_map[dataset].llm_judge(decoded_text,sample["final_answer"], extracted_answer, finish_generation_list[n])
                finally_judge_result = rule_judge_result or llm_judge_result
                judge_info.append({
                    "rule_judge_result": rule_judge_result,
                    "llm_judge_result": llm_judge_result,
                    "finally_judge_result": finally_judge_result
                })
                passat1_list.append(1.0 if finally_judge_result else 0.0)
            
            passat1 = sum(passat1_list)/len(passat1_list)
        elif dataset in CODE_DATASETS:
            k = 1
            judge_info = []
            passat1_list = []
            for n, (decoded_text, finish_generation) in enumerate(zip(decoded_text_list,finish_generation_list)):
                if reeval:
                    if dataset=="humaneval":
                        passat1, single_judge_info = humanevaleval.evaluator_map[dataset].judge(sample["prompt"][0]["value"], decoded_text,  sample["final_answer"], k)
                    elif dataset=="mbpp":
                        passat1, single_judge_info = mbppeval.evaluator_map[dataset].judge(sample["prompt"][0]["value"], decoded_text,  sample["final_answer"], k)
                    elif dataset=="livecodebench":
                        passat1, single_judge_info = 0.0, None
                else:
                    passat1, single_judge_info = 0.0, None
                passat1_list.append(passat1)
                judge_info.append(single_judge_info)
            passat1 = sum(passat1_list)/len(passat1_list)
        else:
            raise ValueError("Unknown dataset: {}".format(dataset))
        
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)
        else:
            print("No results (2)")
            results = []
        done = {entry["idx"] for entry in results}

        if i not in done:
            result = {
                "hyperparams": str(args),
                "prompt": sample["prompt"],
                "completion": decoded_text_list,
                "ground_truth": sample["final_answer"],
                "generated_tokens": [o["meta_info"]["completion_tokens"] for o in outputs],
                "avg_generated_tokens": sum([o["meta_info"]["completion_tokens"] for o in outputs])/len(outputs),
                "time": 0,
                "idx": i,
                "n": len(outputs),
                "finish_generation": finish_generation_list,
                "judge_info": judge_info,
                "passat1_list": passat1_list,
                "passat1": passat1
            }
            results.append(result)
        elif i in done and reeval:
            for result in results:
                if result["idx"] == i:
                    decoded_text = result["completion"]
                    finish_generation = result["finish_generation"]
                    break
            else:
                raise ValueError("No result found for idx: {}".format(i))
            result.update({
                "judge_info": judge_info,
                "passat1_list": passat1_list,
                "passat1": passat1
            })
        with open(results_file, "w") as f:
            results.sort(key=lambda x: x["idx"])
            json.dump(results, f, indent=4)


        total_num = len(results)
        pass_at_1 = sum([r["passat1"] for r in results])
        
        results_statistics = {
            "total_num": total_num,
            "pass@1": pass_at_1 / total_num if total_num > 0 else 0,
            "avg_token_length-all": sum([r["avg_generated_tokens"] for r in results]) / total_num if total_num > 0 else 0,
            "avg_token_length-pass@1>0.75": sum([r["avg_generated_tokens"] for r in results if r["passat1"] > 0.75]) / len([r["passat1"] for r in results if r["passat1"] > 0.75]) if len([r["passat1"] for r in results if r["passat1"] > 0.75]) > 0 else 0,
            "avg_token_length-pass@1>0.5": sum([r["avg_generated_tokens"] for r in results if r["passat1"] > 0.5]) / len([r["passat1"] for r in results if r["passat1"] > 0.5]) if len([r["passat1"] for r in results if r["passat1"] > 0.5]) > 0 else 0,
            "avg_token_length-pass@1>0.25": sum([r["avg_generated_tokens"] for r in results if r["passat1"] > 0.25]) / len([r["passat1"] for r in results if r["passat1"] > 0.25]) if len([r["passat1"] for r in results if r["passat1"] > 0.25]) > 0 else 0,
            "avg_token_length-pass@1>0": sum([r["avg_generated_tokens"] for r in results if r["passat1"] > 0]) / len([r["passat1"] for r in results if r["passat1"] > 0]) if len([r["passat1"] for r in results if r["passat1"] > 0]) > 0 else 0,
        }
        avg_token_length_correct = 0
        count = 0
        for r in results:
            for passat1, generated_tokens in zip(r["passat1_list"], r["generated_tokens"]):
                if passat1==1:
                    avg_token_length_correct += generated_tokens
                    count += 1
        results_statistics["avg_token_length-correct"] = avg_token_length_correct/count if count > 0 else 0

        all_idx = sorted([(r["idx"], r["passat1"]) for r in results], key=lambda x: x[0])
        results_statistics["all_idx"] = {i:j for i,j in all_idx}
        with open(results_statistics_file, "w") as f:
            json.dump(results_statistics, f, indent=4)

        
        if args.push_results_to_hf:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=results_statistics_file,
                path_in_repo=results_statistics_file,
                repo_id=args.hf_repo_id,
                token=args.hf_token
            )

    llm.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
