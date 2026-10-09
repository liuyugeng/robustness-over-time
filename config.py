# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MNLI_LABEL = ['entailment', 'neutral', 'contradiction', 'entailment\'', 'neutral\'', 'contradiction\'']
EQ_LABEL = ['equivalent', 'not_equivalent', 'equivalent\'', 'not_equivalent\'']
ENTAIL_LABEL = ['entailment', 'not_entailment', 'entailment\'', 'not_entailment\'', '0', '1', '0\'', '1\'', 0, 1]

LABEL_SET = {
    # 'positive\'', 'negative\'' is used for label constraint due to a bug of TextAttack repo.
    'sst2': ['positive', 'negative', 'positive\'', 'negative\'', '0', '1', '0\'', '1\'',0, 1], 
    'mnli': MNLI_LABEL,
    'mnli_mismatched': MNLI_LABEL,
    'mnli_matched': MNLI_LABEL,
    'qqp': EQ_LABEL,
    'qnli': ENTAIL_LABEL,
    'rte': ENTAIL_LABEL,
    'cola': ['unacceptable', 'acceptable', 'unacceptable\'', 'acceptable\''],
    'mrpc': EQ_LABEL,
    'wnli': ENTAIL_LABEL,
    'mmlu': ['A', 'B', 'C', 'D', 'A\'', 'B\'', 'C\'', 'D\'', 'a', 'b', 'c', 'd', 'a\'', 'b\'', 'c\'', 'd\''],
    'squad_v2': ['unanswerable', 'unanswerable\''], # do not change the word 'nothing' in prompts.
    'iwslt': ['translate', 'translate\''],
    'un_multi': ['translate', 'translate\''],
    'math': ['math', 'math\''],
    'bool_logic': ['True', 'False', 'True\'', 'False\'', "bool", "boolean", "bool\'", "boolean\'"],
    'valid_parentheses': ['Valid', 'Invalid', 'Valid\'', 'Invalid\'', 'matched', 'matched\'', 'valid', 'invalid', 'valid\'', 'invalid\''],
}

MNLI_LABEL_NEW = ['entailment', 'neutral', 'contradiction']
EQ_LABEL_NEW = ['equivalent', 'not']
ENTAIL_LABEL_NEW = ['entailment', 'not']

LABEL_SET_NEW = {
    # 'positive\'', 'negative\'' is used for label constraint due to a bug of TextAttack repo.
    'sst2': ['positive', 'negative'], 
    'mnli': MNLI_LABEL_NEW,
    'mnli_mismatched': MNLI_LABEL_NEW,
    'mnli_matched': MNLI_LABEL_NEW,
    'qqp': EQ_LABEL_NEW,
    'qnli': ENTAIL_LABEL_NEW,
    'rte': ENTAIL_LABEL_NEW,
    'cola': ['acceptable', 'unacceptable'],
    'mrpc': EQ_LABEL_NEW,
    'wnli': ENTAIL_LABEL_NEW,
    'mmlu': ['A', 'B', 'C', 'D'],
    'squad_v2': ['unanswerable',], # do not change the word 'nothing' in prompts.
}

GENERATE_LEN = {
    'sst2': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 16, 'llama2-13b': 1, 'qwen': 1},
    'mnli': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 16, 'llama2-13b': 1, 'qwen': 1},
    'qqp': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 16, 'llama2-13b': 1, 'qwen': 1},
    'qnli': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 16, 'llama2-13b': 1, 'qwen': 1},
    'rte': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 16, 'llama2-13b': 1, 'qwen': 1},
    'cola': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 16, 'llama2-13b': 1, 'qwen': 1},
    'mrpc': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 16, 'llama2-13b': 1, 'qwen': 1},
    'wnli': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 16, 'llama2-13b': 1, 'qwen': 1},
    'mmlu': {'google/flan-t5-large': 2, 'vicuna-13b': 2, 'google/flan-ul2': 2, "chatgpt": 16, 'llama2-13b': 1, 'qwen': 1},
    'squad_v2': {'google/flan-t5-large': 20, 'google/flan-ul2': 20, "chatgpt": 20, 'llama2-13b': 20},
    'iwslt': {'google/flan-t5-large': 70, 'google/flan-ul2': 70, 'chatgpt': 70, 'llama2-13b': 70},
    'un_multi': {'google/flan-t5-large': 140, 'google/flan-ul2': 140, 'chatgpt': 140, 'llama2-13b': 140},
    'math': {'google/flan-t5-large': 20, 'google/flan-ul2': 20, 'chatgpt': 20, 'llama2-13b': 20},
    'bool_logic': {'google/flan-t5-large': 4,},
}

MODEL_SET = [ 
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0301',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-1106',
    'llama-7b',
    'llama-2-7b',
    'llama-2-7b-chat',
    'llama-13b',
    'llama-2-13b',
    'llama-2-13b-chat',
    'llama-30b',
    'llama-65b',
    'llama-2-70b',
    'llama-2-70b-chat',
    'vicuna-7b-v1.3',
    'vicuna-7b-v1.5',
    'vicuna-13b-v1.3',
    'vicuna-13b-v1.5',
    'chatglm-6b',
    'chatglm2-6b',
    'Baichuan-13B-Base',
    'Baichuan-13B-Chat',
    'Baichuan-7B',
    'Baichuan2-7B-Base',
    'Baichuan2-7B-Chat',
    'Baichuan2-13B-Base',
    'Baichuan2-13B-Chat',
    'gpt-4-turbo-2024-04-09',
    'gpt-4-0125-preview',
    'gpt-4-1106-preview',
    'gpt-4-0613',
    'gpt-4'
    'llama-3-8B',
    'llama-3-70B',
    'llama-3-70B-Instruct',
    'llama-3-8B-Instruct',
    'gpt-4o-2024-11-20',
    'gpt-4o-2024-08-06',
    'gpt-4o-2024-05-13'
]

LABEL_TO_ID = {
    'mmlu': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'},
    'sst2': {'negative': 0, 'positive': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2':2, 0: 0, 1: 1, 2: 2, 'entail': 0,},
    'mnli_mismatched': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2':2, 0: 0, 1: 1, 2: 2, 'entail': 0,},
    'mnli_matched': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2':2, 0: 0, 1: 1, 2: 2, 'entail': 0,},
    'qqp': {'equivalent': 1, 'not_equivalent': 0, '0': 0, '1': 1, 0: 0, 1: 1, 'not': 0},
    'qnli': {'entailment': 0, 'not_entailment': 1, '0': 0, '1': 1, 0: 0, 1: 1, 'entail': 0, 'not entail': 1, 'not': 1},
    'rte': {'entailment': 0, 'not_entailment': 1, '0': 0, '1': 1, 0: 0, 1: 1, 'entail': 0, 'not entail': 1, 'not': 1},
    'cola': {'unacceptable': 0, 'acceptable': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'mrpc': {'equivalent': 1, 'not_equivalent': 0, '0': 0, '1': 1, 0: 0, 1: 1, 'not': 0},
    'wnli': {'entailment': 1, 'not_entailment': 0, '0': 0, '1': 1, 0: 0, 1: 1, 'entails': 1, 'not entail': 0, 'no entailment': 0, 'entail': 1, 'disentail': 0, 'not': 0},
}

ID_TO_LABEL = {
    'mmlu': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'},
    'sst2': {0: 'negative', 1: 'positive'},
    'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mnli_matched': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mnli_mismatched': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'qqp': {1: 'equivalent', 0: 'not_equivalent'},
    'qnli': {0: 'entailment', 1: 'not_entailment'},
    'rte': {0: 'entailment', 1: 'not_entailment'},
    'cola': {0: 'unacceptable', 1: 'acceptable'},
    'mrpc': {1: 'equivalent', 0: 'not_equivalent'},
    'wnli': {1: 'entailment', 0: 'not_entailment'},
}

SUPPORTED_LANGUAGES = {
    'google/flan-t5-large': ['en', 'de', 'fr'],
    'google/flan-ul2': ['en', 'de', 'fr'],
    'vicuna-13b': ['en', 'de', 'fr'],
    'llama2-13b-chat': ['en', 'de', 'fr'],
    'chatgpt': ['en', 'de', 'fr'],
}

LANGUAGES = {
    'ar': 'Arabic', 
    'de': 'German', 
    'en': 'English', 
    'es': 'Spanish',
    'fr': 'French',
    'ru': 'Russian',
    'zh': 'Chinese',
    'it': 'Italian',
    'nl': 'Dutch',
    'ro': 'Romanian',
    'ja': 'Japanese',
    'ko': 'Korean',
}

MATH_QUESTION_TYPES = {
    'algebra_linear_1d': ' linear algebra ',
    'algebra_linear_2d': ' linear algebra ',
    'algebra_sequence_next_term': ' given a sequence predict the next term ',
    'arithmetic_addition_sub_multiple': ' arithmetic addition and subtraction ',
    'arithmetic_mul_div_multiple': ' arithmetic multiplication and division ',
    'arithmetic_mixed': ' arithmetic addition, subtraction, multiplication and division ',
    'arithmetic_nearest_integer_root': ' arithmetic nearest integer root ',
    'comparison_closest': ' compare which one of given numbers is closest to target number ',
    'comparison_kth_biggest': ' compare which one of given numbers is kth biggest or smallest ',
    'comparison_pair': ' comparison which one of given numbers is bigger or smaller ', 
    'measurement_conversion': ' measurement conversion ',
    'numbers_base_conversion': ' numbers base conversion ',
    'numbers_div_remainder': ' numbers division and remainder ',
    'numbers_gcd': ' numbers greatest common divisor ',
    'numbers_is_factor': ' if one number is a factor of antoher number ',
    'number_is_prime': ' if a number is prime ',
    'numbers_lcm': ' least common multiple ',
    'numbers_place_value': ' place value ',
    'numbers_round_number': ' round number ',
    'polynomials_evaluate': ' polynomials evaluate ',
}