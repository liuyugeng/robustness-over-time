import os
import glob
import torch
import socket
if torch.cuda.is_available() and 'juwels' in socket.gethostname():
    os.environ['HF_DATASETS_OFFLINE'] = "1"
    os.environ['TRANSFORMERS_OFFLINE'] = "1"
import logging
import argparse
import datetime
import pandas as pd

from config import *
from check import check_gpt
from parse import retrieve
from inference import Inference

def create_logger(log_path):

    logging.getLogger().handlers = []

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger

def test(logger, inference_model, prompts, write):
    prompt_dict = {}

    if 'gpt' in inference_model.model.lower():
        prompt_dict = inference_model.predict_gpt(prompts, prompt_dict)
    else:
        for prompt in prompts:
            acc = inference_model.predict(prompt)
            prompt_dict[prompt] = acc
            if write:
                logger.info("{:.3f}%, {}\n".format(acc*100, prompt))

            logger.info("=================================================")

    if write:
        values = list(prompt_dict.values())[0] - list(prompt_dict.values())[1]
        logger.info("dropped acc {:.3f}%\n".format(values*100))
    
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--dataset', type=str, default='mnli')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--shot', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--attack', type=str, default='deepwordbug')
    
    parser.add_argument('--arch', type=str, default='chatgpt')

    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_seq_len', type=int, default=20)
    parser.add_argument('--max_batch_size', type=int, default=8)

    args = parser.parse_args()
    return args

def main(args):
    if args.shot == 0:
        prompt_type = "zero"
    elif args.shot == 3:
        prompt_type = "few"
        
    if 'vicuna' in args.model.lower():
        args.generate_len = GENERATE_LEN[args.dataset.split('adv_')[-1]]["vicuna-13b"]
    elif 'qwen' in args.model.lower():
        args.generate_len = GENERATE_LEN[args.dataset.split('adv_')[-1]]["qwen"]
    elif 'llama' in args.model.lower():
        args.generate_len = GENERATE_LEN[args.dataset.split('adv_')[-1]]["llama2-13b"]
    elif 'gpt' in args.model.lower():
        args.generate_len = GENERATE_LEN[args.dataset.split('adv_')[-1]]["chatgpt"]
    else:
        args.generate_len = 20
    args.write = True

    

    # if 'llama' in args.model:
    #     args.write = False
        
    #     if '-2' in args.model:
    #         args.ckpt_dir = os.path.join('/p/project/hai_robtime/models/llama-2', args.model)
    #         args.tokenizer_path = '/p/project/hai_robtime/models/llama-2/tokenizer.model'
    #         args.temperature = 0.6
    #         args.top_p = 0.9
    #         args.max_seq_len = 512
    #         args.max_batch_size = 6
    #     else:
    #         args.ckpt_dir = os.path.join('/p/project/hai_robtime/models/llama', args.model.split('-')[-1].upper())
    #         args.tokenizer_path = '/p/project/hai_robtime/models/llama/tokenizer.model'

    print(args.__dict__)
    
    if args.dataset.split('adv_')[-1] == 'mnli-mm':
        name = 'mnli'
    else:
        name = args.dataset.split('adv_')[-1]
    results = retrieve(args.arch, name, args.attack, prompt_type)
    if results is None:
        exit()
    results = [pd.DataFrame(results).sort_values(by="dropped acc", ascending=False).iloc[0].to_dict()]

    if args.dataset == "iwslt" or args.dataset == "un_multi":
        from config import SUPPORTED_LANGUAGES
        supported_languages = SUPPORTED_LANGUAGES[args.model]

    from dataload import create_dataset
    if args.dataset in ["iwslt", "un_multi"]:
        args.data = create_dataset(args.dataset, args.clean, supported_languages)
    else:
        args.data = create_dataset(args.dataset, args.clean)

    LOGS_DIR = os.path.join("./logs", args.dataset + '/')
    os.makedirs(LOGS_DIR, exist_ok=True)
    today = datetime.date.today()
    prefix = (
        args.model.replace('/', '_')
        + '_' + args.attack
        + "_gen_len_" + str(args.generate_len)
        + "_" + str(args.shot) + "_shot_"
        + args.arch
        + "_dem_" + str(args.clean)
    )

    pattern = os.path.join(LOGS_DIR, prefix + "_*.log")

    for path in glob.glob(pattern):
        os.remove(path)
    # file_name = args.model.replace('/', '_') + '_' + args.attack + "_gen_len_" + str(args.generate_len) + "_" + str(args.shot) + "_shot_" + args.arch + "_dem_" + str(args.clean) + "_" + str(today)
    # if os.path.exists(LOGS_DIR+file_name+".log"):
    #     os.remove(LOGS_DIR+file_name+".log")
    file_name = os.path.join(LOGS_DIR, prefix + "_" + str(today) + ".log")
    logger = create_logger(file_name)
    logger.info(today)
    print(today)
    logger.info(args)

    args.logger = logger

    inference_model = Inference(args)
    for result in results:
        logger.info(result['dropped acc'])
        init_prompt = [result['origin prompt'], result['attack prompt']]
        test(args.logger, inference_model, init_prompt, args.write)
    
if __name__ == '__main__':
    args = get_args()
    #if not check(args):
    # check_gpt(args)
    main(args)
