import transformers
from datasets import load_dataset
import sys
from datetime import datetime
import numpy as np
import outlines
from outlines.samplers import greedy, multinomial, beam_search
import torch
import json
from gsm8k_evals.prompts import prompt_map, standard_prompter
from gsm8k_evals.structure import struct_info
from gsm8k_evals.processing import (
    process_answer, majority_vote, all_pass
    )
from gsm8k_evals import db_tools
import re
import argparse


# should eventually be moved somewhere
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
                        default='standard',
                        choices=list(prompt_map.keys()),
                        help='prompt style to use')
    parser.add_argument('--cot', action=argparse.BooleanOptionalAction, 
                        default=True,
                        help='whether or not to use Chain-of-thought')
    parser.add_argument('--n_shot',
                        default=8,
                        type=int,
                        help="number of examples to use in prompt")
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
    prompter = prompt_map[args.prompt](cot=args.cot, n_shot=args.n_shot)
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
        "cot": args.cot,
        "n_shot": args.n_shot,
        "sampler": args.sampler,
        "n_samples": args.num_samples,
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
        prompt_sample = prompter(dataset[sub_set]['question'][0])
        print("----PROMPT-----")
        print(prompt_sample)
        print("---END PROMPT---")
    else:
        print("----Debugging Regex----")
        print(f"REGEX: {regex_structure}")
        prompt_sample = prompter(dataset[sub_set]['question'][0])
        print("----PROMPT-----")
        print(prompt_sample)
        print("---END PROMPT---")
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
    if args.sampler == "greedy" or args.num_samples == 1:
        test_response = [test_response]
    print("------raw response--")
    print(test_response[0])
    print("------processed response--")
    print(process_response(test_response[0]))
    print("---Running evaluation---")
    last_i = args.i + args.n

    start_t = datetime.now()
    # Main loop
    for start_i in range(args.i,last_i, batch_size):
        end_i = min(start_i+batch_size,last_i)
        prompts = [prompter(dataset[sub_set]['question'][i])
                   for i in range(start_i,end_i)]
        raw_answers = generator(prompts, max_tokens=512)
        # I think this should be considered a bug in outlines
        if batch_size == 1:
            raw_answers = [raw_answers]
        outcomes = []
        for p_i, _ in enumerate(raw_answers):
            i = start_i + p_i
            q_data = {
                'db': db_name,
                'eval_id': eval_id,
                'question_number': i,
                'realized_prompt': prompts[p_i],
                'raw_answer': None,
                'bad_parse': None

            }
            maj = majority_vote(raw_answers[p_i],
                                numeric_answers[i],
                                process_response)
            q_data['maj_correct'] = maj 
            if q_data['maj_correct']:
                outcomes.append('.')
            else:
                outcomes.append('F')
            q_data['pass_correct'] = all_pass(raw_answers[p_i],
                                              numeric_answers[i],
                                              process_response)
            db_tools.add_result(**q_data)
        for outcome in outcomes:
            print(outcome,end='',sep='') 
        if not(i == 0) and ((i % 25) < batch_size):
            print("")
            print(f"[{i}] ",end='',)
            print(db_tools.display_eval_results(db_name, eval_id))
    db_tools.update_evaluation_end(db_name, eval_id)
    print(db_tools.display_eval_results(db_name, eval_id))