import json
import re


# Note: We don't allow leading zeros since 01 is not valid.
# However we're technically not allowing the model to answer
# '0' and none of the answers are 0. Tiny advantage we should keep in mind.
regex_m_300 = r'\{"reasoning": "[\w \d\.\*\-=\+,\?/]{10,300}", "answer": [1-9][0-9]{0,9}\}'
regex_m_400 = r'\{"reasoning": "[\w \d\.\*\-=\+,\?/]{10,400}", "answer": [1-9][0-9]{0,10}\}'
regex_m_100_300 = r'\{"reasoning": "[\w \d\.\*\-=\+,\?/]{100,400}", "answer": [1-9][0-9]{0,10}\}'
regex_m_100_400 = r'\{"reasoning": "[\w \d\.\*\-=\+,\?/]{100,400}", "answer": [1-9][0-9]{0,10}\}'
regex_m_50_500 = r'\{"reasoning": "[\w \d\.\*\-=\+,\?/]{50,500}", "answer": [1-9][0-9]{0,10}\}'
regex_m_50_700 = r'\{"reasoning": "[\w \d\.\*\-=\+,\?/]{50,500}", "answer": [1-9][0-9]{0,10}\}'
regex_hr_300 = r'\{\n    "reasoning": "[\w \d\.\*\-=\+,\?/]{10,300}",\n    "answer": [1-9][0-9]{0,9}\n   \}'
regex_hr_100_400 = r'\{\n    "reasoning": "[\w \d\.\*\-=\+,\?/]{100,400}",\n    "answer": [1-9][0-9]{0,9}\n   \}'
regex_hr_200_400 = r'\{\n    "reasoning": "[\w \d\.\*\-=\+,\?/]{200,400}",\n    "answer": [1-9][0-9]{0,9}\n   \}'
regex_hr_100_500 = r'\{\n    "reasoning": "[\w \d\.\*\-=\+,\?/]{100,500}",\n    "answer": [1-9][0-9]{0,9}\n   \}'
regex_hr_50_400 = r'\{\n    "reasoning": "[\w \d\.\*\-=\+,\?/]{50,400}",\n    "answer": [1-9][0-9]{0,9}\n   \}'
regex_hr_50_500 = r'\{\n    "reasoning": "[\w \d\.\*\-=\+,\?/]{50,500}",\n    "answer": [1-9][0-9]{0,9}\n   \}'
regex_hr_50_700 = r'\{\n    "reasoning": "[\w \d\.\*\-=\+,\?/]{50,700}",\n    "answer": [1-9][0-9]{0,9}\n   \}'

# very similar to the data set range
regex_hr_50_1000 = r'\{\n    "reasoning": "[\w\s\d\.\*\-=+,\?/]{50,1000}",\n    "answer": [1-9][0-9]{0,9}\n   \}'
regex_qa_300 = r'A: [\w \.\*\-=\+,\?/]{10,300}\. The answer is [1-9][0-9]{0,9}\.\n'
regex_qa_50_500 = r'A: [\w \.\*\-=\+,\?/]{50,500}\. The answer is [1-9][0-9]{0,9}\.\n'
regex_qa_50_700 = r'A: [\w \.\*\-=\+,\?/]{50,700}\. The answer is [1-9][0-9]{0,9}\.\n'
regex_qa_alt_1_50_700 = r'Answer - [\w \.\*\-=\+,\?/]{50,700}\. The answer is [1-9][0-9]{0,9}\.'
regex_qa_alt_1_200_700 = r'Answer - [\w \.\*\-=\+,\?/]{200,700}\. The answer is [1-9][0-9]{0,9}\.'
regex_qa_200_700 = r'A: [\w \.\*\-=\+,\?/]{50,700}\. The answer is [1-9][0-9]{0,6}\.\n'

def process_raw_json_response(response):
    return json.loads(response)['answer']

lm_eval_regex = r"The answer is ([1-9][0-9]{0,9})\."
def process_eval_harness(resp):
    # This is what the eval harness uses
    results = re.findall(lm_eval_regex, resp)
    if len(results) > 0:
        val = re.sub(",","",results[0])
        val = float(val)
        return int(val)
    else:
        # a bit of hack, but none of the
        # eval answers are 0.
        return 0

def process_unstruct_json(response):
    try:
        result = json.loads(response.strip('},')).get('answer',0)
    except json.JSONDecodeError:
        result = 0
    except AttributeError:
        result = 0
    return result

struct_info = {
    'unstruct_qa': {
        'regex': None,
        'stop_at': ["Q:", "\n\n"],
        'processor': process_eval_harness
    },
    'unstruct_json': {
        'regex': None,
        'stop_at': ["},"],
        'processor': process_unstruct_json
    },
    'regex_m_300': {
        'regex': regex_m_300,
        'processor': process_raw_json_response
    },
    'regex_m_400': {
        'regex': regex_m_400,
        'processor': process_raw_json_response
    },
    'regex_m_50_500': {
        'regex': regex_m_50_500,
        'processor': process_raw_json_response
    },
    'regex_m_50_700': {
        'regex': regex_m_50_700,
        'processor': process_raw_json_response
    },
    'regex_hr_300': {
        'regex': regex_hr_300,
        'processor': process_raw_json_response
    },
    'regex_hr_50_400':{
        'regex': regex_hr_50_400,
        'processor': process_raw_json_response
    },
    'regex_hr_100_400':{
        'regex': regex_hr_100_400,
        'processor': process_raw_json_response
    },
    'regex_hr_200_400':{
        'regex': regex_hr_200_400,
        'processor': process_raw_json_response
    },
    'regex_hr_100_500':{
        'regex': regex_hr_100_500,
        'processor': process_raw_json_response
    },
    'regex_hr_50_500':{
        'regex': regex_hr_50_500,
        'processor': process_raw_json_response
    },
    'regex_hr_50_700':{
        'regex': regex_hr_50_700,
        'processor': process_raw_json_response
    },
    'regex_hr_50_1000':{
        'regex': regex_hr_50_1000,
        'processor': process_raw_json_response
    },
    'regex_qa_300': {
        'regex': regex_qa_300,
        'processor': process_eval_harness
    },
    'regex_qa_50_500': {
        'regex': regex_qa_50_500,
        'processor': process_eval_harness
    },
    'regex_qa_50_700': {
        'regex': regex_qa_50_700,
        'processor': process_eval_harness
    },
    'regex_qa_200_700': {
        'regex': regex_qa_200_700,
        'processor': process_eval_harness
    },
    'regex_qa_alt_1_200_700': {
        'regex': regex_qa_alt_1_200_700,
        'processor': process_eval_harness
    },
    'regex_qa_alt_1_50_700': {
        'regex': regex_qa_alt_1_50_700,
        'processor': process_eval_harness
    }
}