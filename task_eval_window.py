import pwd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
from tqdm import tqdm
import argparse
import numpy as np
import os
from accelerate import Accelerator
from accelerate.utils import set_seed
# from matheval import evaluator_map, set_client, AIMEEvaluator
import matheval
import humanevaleval
import mbppeval

accelerator = Accelerator()
MATH_DATASETS = ["math500","aime2024","gpqa_diamond","gsm8k"]
CODE_DATASETS = ["humaneval","mbpp","livecodebench"]

def my_set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # If using accelerate, set the seed for it as well
    try:
        set_seed(seed_value)
    except ImportError:
        pass  # accelerate is not installed, continue without it

# Create a parser for command line arguments
parser = argparse.ArgumentParser(description='Process some parameters for text generation.')

parser.add_argument('--dataset', type=str, choices=["math500","aime2024","gpqa_diamond","gsm8k","humaneval","mbpp","livecodebench"], help='Name of dataset')
parser.add_argument('--model_name', type=str, default=None, help='Model name')
parser.add_argument('--seed', type=int, default=0, help='Seed value for reproducibility')
parser.add_argument('--max_generated_tokens', type=int, default=10000, help='Limit the number of generated tokens')
parser.add_argument('--continuous_thinking', action='store_true', help='Enable continuous thinking')
parser.add_argument('--gumble_softmax', action='store_true', help='Enable gumble softmax for think logits')
parser.add_argument('--force_end_thinking', action='store_true', help='If force to end thinking')
parser.add_argument('--force_end_repeating', action='store_true', help='If force to end repeating')
parser.add_argument('--force_end_repeating_after', type=int, default=100, help='Force to end repeating after n token too confident')
parser.add_argument('--force_end_repeating_threshold', type=float, default=0.99, help='Threshold to force to end repeating after n token too confident')
parser.add_argument('--force_end_repeating_prompt', type=str, default="\n</think>\n\n", help='Threshold to force to end repeating after n token too confident')
parser.add_argument('--continuous_window_size', type=int, default=None, help='Window size for continuous thinking')
parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling probability')
parser.add_argument('--repetition_penalty', type=float, default=0.0, help='Sampling temperature')
parser.add_argument('--repetition_window', type=int, default=1000000, help='')
parser.add_argument('--repetition_penalty_start_idx', type=int, default=1000000, help='')
parser.add_argument('--think_temperature', type=float, default=0.9, help='Thinking temperature for generation')
parser.add_argument('--think_top_p', type=float, default=0.9, help='Top-p for thinking sampling')
parser.add_argument('--think_top_k', type=int, default=4, help='Top-k sampling number')
parser.add_argument('--gumble_epsilon', type=float, default=0.2, help='')
parser.add_argument('--gumble_factor', type=float, default=0.5, help='')
parser.add_argument('--prob_cutoff', type=float, default=0.05, help='Cutoff probability')
parser.add_argument('--cutoff_add', type=float, default=0.1, help='add cutoff probability')
parser.add_argument('--t_divide', type=float, default=1.0, help='add cutoff probability')
parser.add_argument('--cutoff_add_per_len', type=int, default=4000, help='add cutoff probability per length')
parser.add_argument('--additional_tokens_after_thinking', type=int, default=2000, help='additional number of tokens after force stop thinking.')
parser.add_argument('--max_cutoff', type=float, default=0.25, help='add cutoff probability per length')
parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing samples')
parser.add_argument('--end_idx', type=int, default=500, help='End index for processing samples')
parser.add_argument('--idx_list', type=str, default=None, help='index for processing samples')
parser.add_argument('--no_print_decoding', action='store_true', help='noEnable printing while decoding')
parser.add_argument('--kvcache', action='store_true', help='Enable kv cache')
parser.add_argument('--flashattn', action='store_true', help='Enable flashattn')
parser.add_argument('--reeval', action='store_true', help='Enable re-evaluation')
parser.add_argument('--api_base', type=str, default=None, help='')
parser.add_argument('--deployment_name', type=str, default=None, help='')
parser.add_argument('--api_version', type=str, default=None, help='')
parser.add_argument('--api_key', type=str, default=None, help='')

parser.add_argument('--push_results_to_hf', action='store_true', help='Enable push to huggingface')
parser.add_argument('--hf_token', type=str, default=None, help='')
parser.add_argument('--hf_repo_id', type=str, default=None, help='')


# Parse the arguments
args = parser.parse_args()

matheval.set_client(args.api_base, args.deployment_name, args.api_version, args.api_key)


# Assign variables from command line arguments
dataset = args.dataset
max_generated_tokens = args.max_generated_tokens
seed = args.seed
continuous_thinking = args.continuous_thinking
gumble_softmax = args.gumble_softmax
org_force_end_thinking = args.force_end_thinking
org_force_end_repeating = args.force_end_repeating
force_end_repeating_after = args.force_end_repeating_after
force_end_repeating_threshold = args.force_end_repeating_threshold
force_end_repeating_prompt = args.force_end_repeating_prompt

continuous_window_size = args.continuous_window_size
temperature = args.temperature
top_p = args.top_p
repetition_penalty = args.repetition_penalty
repetition_window = args.repetition_window
repetition_penalty_start_idx = args.repetition_penalty_start_idx
org_think_temperature = args.think_temperature
think_top_p = args.think_top_p
think_top_k = args.think_top_k
gumble_epsilon = args.gumble_epsilon
gumble_factor = args.gumble_factor
org_prob_cutoff = args.prob_cutoff
cutoff_add = args.cutoff_add
t_divide = args.t_divide
cutoff_add_per_len = args.cutoff_add_per_len
additional_tokens_after_thinking = args.additional_tokens_after_thinking
max_cutoff = args.max_cutoff
start_idx = args.start_idx
end_idx = args.end_idx
idx_list = args.idx_list
no_print_decoding = args.no_print_decoding
model_name = args.model_name
kvcache = args.kvcache
flashattn = args.flashattn
reeval = args.reeval

print(args)


# Modle and Tokenizer
# my_device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 3, 'model.layers.9': 3, 'model.layers.10': 3, 'model.layers.11': 3, 'model.layers.12': 4, 'model.layers.13': 4, 'model.layers.14': 4, 'model.layers.15': 4, 'model.layers.16': 5, 'model.layers.17': 5, 'model.layers.18': 5, 'model.layers.19': 5, 'model.layers.20': 6, 'model.layers.21': 6, 'model.layers.22': 6, 'model.layers.23': 6, 'model.layers.24': 7, 'model.layers.25': 7, 'model.layers.26': 7, 'model.layers.27': 7, 'model.norm': 7, 'model.rotary_emb': 7, 'lm_head': 2}
my_device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
print(tokenizer.eos_token)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, attn_implementation="flash_attention_2" if flashattn else None, torch_dtype=torch.bfloat16, device_map=my_device_map)
print(model.hf_device_map)
model.eval()  # 设置为评估模式
if kvcache:
    print("enable kvcache")
    model.config.use_cache = True  # 启用 KV 缓存
    model.config.use_flash_attn = True  # 启用 Flash Attention（如果模型支持）

# Load the JSON file
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

def apply_repetition_penalty(logits, input_ids, penalty_factor=2.0):
    """
    Apply repetition penalty to the logits. The penalty increases the probability of previously generated tokens.
    """
    for i in range(input_ids.shape[1]):  # Iterate over the sequence
        token_id = input_ids[0, i].item()
        logits[token_id] /= penalty_factor  # Apply penalty to already generated tokens
    return logits

os.makedirs(f"results", exist_ok=True)
results_file = f"results/{model_name.split('/')[-1]}_{dataset}_ct{continuous_thinking}_{org_think_temperature}_{think_top_p}_{think_top_k}_{t_divide}_{org_prob_cutoff}_{cutoff_add}_{cutoff_add_per_len}.json"
results_statistics_file = f"results/{model_name.split('/')[-1]}_{dataset}_ct{continuous_thinking}_{org_think_temperature}_{think_top_p}_{think_top_k}_{t_divide}_{org_prob_cutoff}_{cutoff_add}_{cutoff_add_per_len}_statistics.json"

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
else:
    print("No results (1)")
    results = []
done = {entry["idx"] for entry in results}

if not idx_list:
    bar = tqdm(zip(range(start_idx,min(end_idx,len(samples))),samples[start_idx:min(end_idx,len(samples))]),total=min(end_idx,len(samples))-start_idx)
else:
    idx_list = [int(idx) for idx in idx_list.split(",")]
    bar = tqdm(zip(idx_list,[samples[idx] for idx in idx_list]),total=len(idx_list))

with torch.no_grad():
    for idx,sample in bar:

        if idx not in done:
            my_set_seed(seed)

            # reset
            prob_cutoff = org_prob_cutoff
            think_temperature = org_think_temperature
            # flags
            if continuous_thinking:
                think = True
            else:
                think = False
            finish_generation = False
            force_end_thinking = False
            force_end_repeating_count = 0
            force_end_repeating_times = 0
            thinking_len = -1

            # Initialize the chat template
            if dataset == "aime2024" or dataset == "math500" or dataset == "gsm8k":
                chat = [
                        {
                            "role": "user",
                            "content": MATH_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])
                        }
                    ]
            elif dataset == "gpqa_diamond":
                chat = [
                        {
                            "role": "user",
                            "content": GPQA_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])
                        }
                    ]
            elif dataset == "humaneval":
                chat = [
                        {
                            "role": "user",
                            "content": CODE_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])
                        }
                    ]
            elif dataset == "mbpp":
                chat = [
                        {
                            "role": "user",
                            "content": MBPP_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"], TestCases="\n".join(sample["final_answer"]["test_list"]))
                        }
                    ]
            elif dataset == "livecodebench":
                chat = [
                        {
                            "role": "user",
                            "content": get_lcb_prompt(question_content=sample["prompt"][0]["value"], starter_code=sample["final_answer"]["starter_code"])
                        }
                    ]
            else:
                raise ValueError("Invalid dataset name")

            # Add the chat template to the prompts for generation.
            prompts = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            print("Template prompt:", prompts,flush=True)
            input_ids = tokenizer.encode(prompts, return_tensors="pt", add_special_tokens=False).to(model.device)
            inputs_embeds = model.get_input_embeddings()(input_ids)
            true_inputs_embeds = inputs_embeds
            past_key_values = None

            # Generate a pair of hidden states and logits at a time
            # Then construct soft thinking token (not a hard token in vocabulary, but a weighted sum of all embeddings)
            # This token contain the information of all brank of next token, which is a implicit tree search
            # Finally concatenate the new embedding to the input_embeds
            # We skip the tokenization while thinking
            start = time.time()
            i = 0
            while True:
                bar.set_description(f"idx: {idx}, Tokens generating: {i}")
                
                if (i+1) % cutoff_add_per_len == 0:
                    think_temperature /= t_divide
                    # prob_cutoff += 0.1
                    # prob_cutoff = min(prob_cutoff, 0.25)
                    prob_cutoff += cutoff_add
                    prob_cutoff = min(prob_cutoff, max_cutoff)
                    # think = False
                    pass
                # Call model.forward() to get the hidden states and logits
                outputs = model(inputs_embeds=true_inputs_embeds,
                                output_hidden_states=True,
                                use_cache=args.kvcache,
                                past_key_values=past_key_values
                                )
                if args.kvcache:
                    past_key_values = outputs.past_key_values
                    pass
                hidden_states = outputs.hidden_states
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                # if repetition_penalty > 0 and i>repetition_penalty_start_idx:
                #     next_token_logits = apply_repetition_penalty(next_token_logits, input_ids[:,-repetition_window:], penalty_factor=repetition_penalty)

                # For soft thinking, we still generate a token greedily just for visualization, which will not be used in input_ids for next step
                # Then calculate the soft thinking token by weighted sum of embeddings
                # After soft thinking, this is the same as top p sampling with temperature
                if think:
                    # next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                    scaled_logits = next_token_logits / think_temperature
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)


                    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)


                    sorted_logits_, sorted_indices_ = torch.sort(next_token_logits, descending=True)
                    sorted_probs_ = torch.nn.functional.softmax(sorted_logits_, dim=-1)
                    if sorted_probs_[0] > force_end_repeating_threshold:
                        force_end_repeating_count += 1
                    else:
                        force_end_repeating_count = 0

                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = (cumulative_probs > think_top_p) | (sorted_probs < prob_cutoff)
                    sorted_indices_to_remove[..., 0] = 0

                    # Apply mask to logits
                    sorted_logits = sorted_logits[~sorted_indices_to_remove][:think_top_k]
                    sorted_indices = sorted_indices[~sorted_indices_to_remove][:think_top_k]

                    # sorted_logits[sorted_indices_to_remove] = -float('Inf')
                    # Apply softmax to get probabilities
                    if gumble_softmax:
                        u = torch.clamp(torch.rand(sorted_logits.shape, device=sorted_logits.device,dtype=sorted_logits.dtype), gumble_epsilon, 1 - gumble_epsilon)
                        gumbel_noise = -torch.log(-torch.log(u)) 
                        gumbel_noise = gumble_factor * gumbel_noise
                        gumbels = (sorted_logits + gumbel_noise)
                        sorted_probs = torch.nn.functional.softmax(gumbels, dim=-1)
                    else:
                        sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)

                    next_token_id = sorted_indices[torch.argmax(sorted_probs, dim=-1)].unsqueeze(0)

                    print(len(sorted_logits), sorted_probs.tolist(), [tokenizer.decode(i, skip_special_tokens=False) for i in sorted_indices],flush=True)

                    new_hidden_states = (model.get_input_embeddings().weight[sorted_indices].T @ sorted_probs).T.unsqueeze(0).unsqueeze(0)
                else:
                    scaled_logits = next_token_logits / temperature
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)

                    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)


                    sorted_logits_, sorted_indices_ = torch.sort(next_token_logits, descending=True)
                    sorted_probs_ = torch.nn.functional.softmax(sorted_logits_, dim=-1)
                    if sorted_probs_[0] > force_end_repeating_threshold:
                        force_end_repeating_count += 1
                    else:
                        force_end_repeating_count = 0

                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Apply mask to logits
                    sorted_logits[sorted_indices_to_remove] = -float('Inf')
                    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)

                    index = torch.multinomial(sorted_probs, num_samples=1)
                    next_token_id = sorted_indices[index]
                    print(tokenizer.decode(next_token_id, skip_special_tokens=False), flush=True)
                    
                # concatenate the new token to the input_ids
                next_token_id = next_token_id.unsqueeze(0)  # 调整维度为 [batch_size, 1]
                input_ids = torch.cat([input_ids, next_token_id], dim=1)

                generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                if (i+1)%1000==0 and (not no_print_decoding):
                    print("Generated text: ", generated_text,"|",sep="",flush=True)
                    print(i,flush=True)

                if think and generated_text.count("</think>") == 1:
                    if (not no_print_decoding):
                        print("stop thinking",flush=True)
                    think = False
                    thinking_len = i+1

                if think:
                    # print(new_hidden_states)
                    inputs_embeds = torch.cat([inputs_embeds,new_hidden_states], dim=1)
                else:
                    new_token_embedding = model.get_input_embeddings()(next_token_id)
                    inputs_embeds = torch.cat([inputs_embeds,new_token_embedding], dim=1)
                
                if continuous_window_size is not None:
                    inputs_embeds_1 = model.get_input_embeddings()(input_ids[:, :-continuous_window_size])
                    inputs_embeds_2 = inputs_embeds[:, -continuous_window_size:]
                    true_inputs_embeds = torch.cat([inputs_embeds_1, inputs_embeds_2], dim=1)
                else:
                    true_inputs_embeds = inputs_embeds
                
                if kvcache:
                    true_inputs_embeds = true_inputs_embeds[:,-1:,:]

                if tokenizer.eos_token_id == next_token_id.item() :
                    if i < max_generated_tokens:
                        finish_generation = True
                    break

                if i == max_generated_tokens-1 :
                    if org_force_end_thinking:
                        if generated_text.count("</think>") == 0:
                            print("Force end thinking")
                            force_end_thinking = True
                            think = False
                            stop_think_ids = tokenizer.encode("\n</think>", return_tensors="pt", add_special_tokens=False).to(model.device)
                            stop_think_inputs_embeds = model.get_input_embeddings()(stop_think_ids)
                            inputs_embeds = torch.cat([inputs_embeds, stop_think_inputs_embeds], dim=1)
                            input_ids = torch.cat([input_ids, stop_think_ids], dim=1)
                            if kvcache:
                                true_inputs_embeds = inputs_embeds[:,-1-stop_think_inputs_embeds.shape[1]:,:]
                                print(true_inputs_embeds.shape,-1-stop_think_inputs_embeds.shape[1])
                            else:
                                true_inputs_embeds = inputs_embeds
                        else:
                            think = False
                    else:
                        break

                if org_force_end_repeating and force_end_repeating_count >= force_end_repeating_after and force_end_repeating_times<=1:
                    print("Force end repeating")
                    force_end_repeating_times += 1
                    force_end_repeating_count = 0
                    if think:
                        force_end_repeating_ids = tokenizer.encode(force_end_repeating_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                        force_end_repeating_inputs_embeds = model.get_input_embeddings()(force_end_repeating_ids)
                        inputs_embeds = torch.cat([inputs_embeds[:,:-force_end_repeating_after], force_end_repeating_inputs_embeds], dim=1)
                        input_ids = torch.cat([input_ids[:,:-force_end_repeating_after*2], force_end_repeating_ids], dim=1)
                        
                        past_key_values = None
                        torch.cuda.empty_cache()
                    else:
                        force_end_repeating_ids = tokenizer.encode(tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).to(model.device)
                        force_end_repeating_inputs_embeds = model.get_input_embeddings()(force_end_repeating_ids)
                        inputs_embeds = torch.cat([inputs_embeds, force_end_repeating_inputs_embeds], dim=1)
                        input_ids = torch.cat([input_ids, force_end_repeating_ids], dim=1)
                        
                        past_key_values = None
                        torch.cuda.empty_cache()
                        break

                    # if kvcache:
                    #     true_inputs_embeds = inputs_embeds[:,-1-force_end_repeating_inputs_embeds.shape[1]:,:]
                    #     print(true_inputs_embeds.shape,-1-force_end_repeating_inputs_embeds.shape[1])
                    # else:
                    true_inputs_embeds = inputs_embeds


                if i >= max_generated_tokens+additional_tokens_after_thinking:
                    break
                i+=1
                    
            decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print("Generated text: ", decoded_text, flush=True)
            print("Time: ", i+1, flush=True)

            end = time.time()
            print("生成时间：", end - start, flush=True)
        elif reeval:
            for result in results:
                if result["idx"] == idx:
                    decoded_text = result["completion"]
                    finish_generation = result["finish_generation"]
                    break
            else:
                raise ValueError("No result found for idx: {}".format(idx))

        else:
            continue

        if dataset in MATH_DATASETS:
            rule_judge_result = None
            rule_judge_result, extracted_answer = matheval.evaluator_map[dataset].rule_judge(decoded_text,sample["final_answer"], finish_generation)
            llm_judge_result = None
            if not rule_judge_result:
                llm_judge_result = matheval.evaluator_map[dataset].llm_judge(decoded_text,sample["final_answer"], extracted_answer, finish_generation)
            finally_judge_result = rule_judge_result or llm_judge_result
            judge_info = {
                "rule_judge_result": rule_judge_result,
                "llm_judge_result": llm_judge_result,
                "finally_judge_result": finally_judge_result
            }
            passat1 = 1.0 if finally_judge_result else 0.0
        elif dataset in CODE_DATASETS:
            k = 1
            if dataset=="humaneval":

                passat1, judge_info = humanevaleval.evaluator_map[dataset].judge(sample["prompt"][0]["value"], decoded_text,  sample["final_answer"], k)
            elif dataset=="mbpp":
                passat1, judge_info = mbppeval.evaluator_map[dataset].judge(sample["prompt"][0]["value"], decoded_text,  sample["final_answer"], k)
            elif dataset=="livecodebench":
                passat1, judge_info = 0.0, None
        else:
            raise ValueError("Unknown dataset: {}".format(dataset))
            

        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)
        else:
            print("No results (2)")
            results = []
        done = {entry["idx"] for entry in results}

        if idx not in done:


            results.append({
                "hyperparams": str(args),
                "prompt": sample["prompt"],
                "completion": decoded_text,
                "ground_truth": sample["final_answer"],
                "generated_tokens": i+1,
                "think_len": thinking_len,
                "finish_generation": finish_generation,
                "force_end_thinking": force_end_thinking,
                "force_end_repeating_times": force_end_repeating_times,
                "time": end - start,
                "idx": idx,
                "judge_info": judge_info,
                "passat1": passat1
            })
        elif idx in done and reeval:
            for result in results:
                if result["idx"] == idx:
                    decoded_text = result["completion"]
                    finish_generation = result["finish_generation"]
                    break
            else:
                raise ValueError("No result found for idx: {}".format(idx))
            result.update({
                "judge_info": judge_info,
                "passat1": passat1
            })
                



        with open(results_file, "w") as f:
            results.sort(key=lambda x: x["idx"])
            json.dump(results, f, indent=4)

        total_num = len(results)
        pass_at_1 = sum([r["passat1"] for r in results]) / total_num if total_num > 0 else 0
        avg_token_length_all = sum([r["generated_tokens"] for r in results]) / total_num if total_num > 0 else 0
        avg_token_length_correct = sum([r["generated_tokens"] for r in results if r["passat1"] > 0.0]) / len([r["passat1"] for r in results if r["passat1"] > 0.0]) if len([r["passat1"] for r in results if r["passat1"] > 0.0]) > 0 else 0
        
        all_idx = sorted([(r["idx"], r["passat1"]) for r in results], key=lambda x: x[0])
        results_statistics = {
            "total_num": total_num,
            "pass@1": pass_at_1,
            "avg_token_length-all": avg_token_length_all,
            "avg_token_length-correct": avg_token_length_correct,
            "all_idx": {i:j for i,j in all_idx}
        }
        with open(results_statistics_file, "w") as f:
            json.dump(results_statistics, f, indent=4)

        if args.push_results_to_hf:

            from huggingface_hub import HfApi
            try:
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=results_statistics_file,
                    path_in_repo=results_statistics_file,
                    repo_id=args.hf_repo_id,
                    token=args.hf_token
                )
                api.upload_file(
                    path_or_fileobj=results_file,
                    path_in_repo=results_file,
                    repo_id=args.hf_repo_id,
                    token=args.hf_token
                )
            except:
                print("fail to push to huggingface")
