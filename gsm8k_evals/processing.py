import re
from collections import Counter


def process_answer(raw_answer):
    answer_string = raw_answer.split("#### ")[-1]
    # Note: they do NOT do this cleanup
    commas_removed = re.sub(",","",answer_string)
    return int(commas_removed)

def apply_processor(results, processor):
    return [processor(result) for result in results]

# Maj @
# Note, this may be biased towards cases where many exampls fail
# to process, this can be solved by filtering those.
def majority_vote(results, answer, processor=process_answer):
    if not isinstance(results,list):
        results = [results]
    processed_results = apply_processor(results, processor)
    model_answer = Counter(processed_results).most_common(1)[0][0]
    return model_answer == answer

def all_pass(results, answer, processor=process_answer):
    processed_results = apply_processor(results, processor)
    return any([
        result == answer for result in processed_results
    ])