import transformers
from datasets import load_dataset
import sys
from datetime import datetime
import numpy as np
import outlines
from outlines.samplers import greedy, multinomial, beam_search
import torch
import json
from gsm8k_evals.prompts import prompt_map
from gsm8k_evals.structure import struct_info
from gsm8k_evals import db_tools
import re
import argparse

# this can be moved some place one we pull out the datasets
def process_answer(raw_answer):
    answer_string = raw_answer.split("#### ")[-1]
    # Note: they do NOT do this cleanup
    commas_removed = re.sub(",","",answer_string)
    return int(commas_removed)

# should eventually be moved somewhere
# need to be updated with the n_samples
samplers = {
    'greedy': lambda n_samples: greedy(),
    # update these to use different k etc.
    'multinomial': lambda n_samples: multinomial(samples=n_samples),
    'm8': lambda n_samples: multinomial(top_k=8, samples=n_samples),
    'm4': lambda n_samples: multinomial(top_k=4, samples=n_samples),
    'm2': lambda n_samples: multinomial(top_k=2, samples=n_samples),
    'beam': lambda n_samples: beam_search(beams=n_samples),
}

if __name__ == "__main__":
    print(transformers.__version__)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n',
                        dest='n',
                        default=5,
                        type=int,
                        help='number of samples')
    ## remove this eventually?
    parser.add_argument('-i',
                        dest='i',
                        default=0,
                        type=int,
                        help='index to start sampling')
    parser.add_argument('-b',
                        dest='b',
                        default=1,
                        type=int,
                        help='specify batch size')
    parser.add_argument('--prompt',
                        dest='prompt',
                        default='standard_8',
                        choices=list(prompt_map.keys()),
                        help='prompt style to use')
    parser.add_argument('--struct',
                        dest='struct',
                        default='unstruct_qa',
                        choices=list(struct_info.keys()),
                        help='structure the model will adhere to')
    parser.add_argument('--model',
                        dest='model_name',
                        default='mistralai/Mistral-7B-v0.1',
                        help='model to use for generation'
                        )
    parser.add_argument('--device',
                        dest='device',
                        default='mps',
                        choices=['mps','cuda','cpu'],
                        help='device to run the model on')
    parser.add_argument('--sampler',
                        dest='sampler',
                        default='greedy',
                        choices=list(samplers.keys()),
                        help="selects sampler to use during generation"
                        )
    parser.add_argument('--num_samples',
                        default=1,
                        type=int,
                        help="number of samples used by sampler")
    parser.add_argument('--db',
                        dest='db_name',
                        default='results.db',
                        help='sqlite database for storing results')
    parser.add_argument('--sub_set',
                        dest='sub_set',
                        default='test',
                        choices=['test','train'],
                        help='specify test or train set'
                        )
    
    args = parser.parse_args()
    device=args.device
    prompter = prompt_map[args.prompt]
    regex_structure = struct_info[args.struct]['regex']
    process_response = struct_info[args.struct]['processor']
    model_name = args.model_name
    sampler = samplers[args.sampler](args.num_samples)
    db_name = args.db_name
    sub_set = args.sub_set
    batch_size = args.b
    
    db_tools.create_evaluation_table(db_name)
    db_tools.create_result_table(db_name)
    eval_args = {
        "db": db_name,
        "model": model_name,
        "dataset": "gsm8k",
        "sub_set": sub_set,
        "start_time": datetime.now(),
        "sampler": args.sampler,
        "prompt_name": args.prompt,
        "struct_name": args.struct,
    }
    eval_id = db_tools.add_evaluation(**eval_args)

    print("Loading dataset...")
    dataset = load_dataset("gsm8k", "main")
    numeric_answers = [process_answer(answer.split("#### ")[-1])
            for answer in dataset[sub_set]['answer']]
        
    print(f"Test questions: {len(dataset[sub_set])}")
    if regex_structure is None:
        print("performing unstructured generation")
    else:
        print("----Debugging Regex----")
        prompt_sample = prompter(dataset[sub_set]['question'][0])
        print(f"REGEX: {regex_structure}")
        print("Testing regex (should find 8 samples):")
        regex_found = re.findall(regex_structure,prompt_sample)
        print(f"Found {len(regex_found)}/8")
        for found in regex_found:
            print(found)
    print("---Loading Model----")
    model = outlines.models.transformers(
        model_name,
        device=device,
        model_kwargs={
            'torch_dtype': torch.float16,
            #'trust_remote_code': True
        },
        tokenizer_kwargs={
            'torch_dtype': torch.float16,
            #'trust_remote_code': True
        }
        )
    print("---Building Generator---")
    if regex_structure is None:
        stop_str = struct_info[args.struct]['stop_at']
        generator = outlines.generate.text(model,
                                        sampler=sampler)
    else:
        generator = outlines.generate.regex(
            model,
            regex_structure,
            sampler=sampler)
    print("---Sampling from Generator---")
    if regex_structure is None:
        stop_str = struct_info[args.struct]['stop_at']
        test_response = generator(
            prompter(dataset[sub_set]['question'][19]),
            stop_at=stop_str,
            max_tokens=512)
    else:
        test_response = generator(prompter(dataset[sub_set]['question'][19]), 
                            max_tokens=512)
    print("------raw response--")
    print(test_response)
    print("------processed response--")
    print(process_response(test_response))
    print("---Running evaluation---")
    last_i = args.i + args.n

    start_t = datetime.now()
    # need to update this to work with batches.
    for start_i in range(args.i,last_i, batch_size):
        end_i = min(start_i+batch_size,last_i)
        prompts = [prompter(dataset[sub_set]['question'][i])
                   for i in range(start_i,end_i)]
        raw_answers = generator(prompts, max_tokens=512)
        # I think this should be considered a bug in outlines
        if batch_size == 1:
            raw_answers = [raw_answers]
        for p_i, _ in enumerate(prompts):
            i = start_i + p_i
            q_data = {
                'db': db_name,
                'eval_id': eval_id,
                'question_number': i,
                'start_time': datetime.now()
            }
            q_data['realized_prompt'] = prompts[p_i]
            q_data['raw_answer'] = raw_answers[p_i]
            try:
                q_data['answer'] = process_response(q_data['raw_answer'])
                q_data['bad_parse'] = False
            except json.JSONDecodeError as e:
                print(f"error at q:{i}")
                print(q_data['raw_answer'])
                q_data['answer'] = 0
                q_data['bad_parse'] = True
            # just a heuristic    
            if (regex_structure is None) and (q_data['answer'] == 0):
                q_data['bad_parse'] = True
        
            q_data['correct'] = q_data['answer'] == numeric_answers[i]
            if q_data['correct']:
                print('.',end='')
            elif q_data['bad_parse']:
                print('X',end='')
            else:
                print('F',end='')
            # time is a bit off now for batching
            # this should eventually be cleaned up.
            q_data['end_time'] = datetime.now()
            db_tools.add_result(**q_data)
            sys.stdout.flush()
        # this won't work right with batching    
        if not(i == 0) and (i % 25 == 0):
            print("")
            print(f"[{i}] ",end='')
            print(db_tools.display_eval_results(db_name, eval_id))
        print("")
    db_tools.update_evaluation_end(db_name, eval_id)
    print(db_tools.display_eval_results(db_name, eval_id))