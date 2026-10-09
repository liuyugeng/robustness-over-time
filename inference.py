# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import io
import sys
import json
import time
import torch
import inspect
from openai import OpenAI
import fastchat
from config import LABEL_SET, LABEL_TO_ID, LABEL_SET_NEW
from tqdm import tqdm, trange
from threading import Thread

from typing import Tuple, List, Literal, Optional, Tuple, TypedDict, Iterator
from pathlib import Path
from llama1 import ModelArgs, Transformer, Tokenizer, LLaMA
from llama2 import Llama
from fastchat.model import load_model, get_conversation_template
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from transformers.generation.utils import GenerationConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, TextIteratorStreamer, LlamaForCausalLM, AutoModel, TextStreamer, pipeline, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

"""
This clss implements the inference of the model (including create the model).
"""
class Inference(object):

    def __init__(self, args):
        self.error_analysis = False
        self.args = args
        self.model = args.model
        self.loading(self.model)

    def process_input(self, prompt, raw_data):
        if self.args.dataset in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli", "adv_sst2", "adv_qqp", "adv_mnli", "adv_mnli-mm", "adv_qnli", "adv_rte"]:
            return self._process_cls_input(prompt, raw_data)
        elif self.args.dataset == "mmlu":
            return self._process_qa_input(prompt, raw_data)
        elif self.args.dataset == "squad_v2":
            return self._process_squad_v2_input(prompt, raw_data)
        elif self.args.dataset in ['iwslt', 'un_multi']:
            return self._process_trans_input(prompt, raw_data)
        elif self.args.dataset == 'math':
            return self._process_math_input(prompt, raw_data)
        else:
            raise NotImplementedError("The dataset is not implemented!")
    

    def process_pred(self, pred):
        if self.args.dataset in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli", "adv_sst2", "adv_qqp", "adv_mnli", "adv_mnli-mm", "adv_qnli", "adv_rte"]:
            return self._process_cls_pred(pred)
        elif self.args.dataset == "mmlu":
            return self._process_qa_pred(pred)
        elif self.args.dataset == "squad_v2":
            return self._process_squad_v2_pred(pred)
        elif self.args.dataset in ['iwslt', 'un_multi']:
            return self._process_trans_pred(pred)
        elif self.args.dataset == 'math':
            return self._process_math_pred(pred)
        else:
            raise NotImplementedError("The dataset is not implemented!")
        
    def loading(self, model):
        if 'llama' in model:
            if 'llama-3' in model.lower():
                model_dir = 'meta-llama/Meta-L' + model[1:]
            elif 'llama-2' in model.lower():
                model_dir = 'meta-llama/L' + model[1:] + '-hf'
            else:
                import socket
                if 'juwels' in socket.gethostname():
                    model_dir = os.path.join('/p/project/hai_aaac/models/hf_llama',model)
                else:
                    model_dir = os.path.join('/data/models', model)

            # self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
            # self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
            # if 'llama-3' in model.lower() and 'instruct' not in model.lower():
            if 'llama-3' in model.lower():
                self.pipe = pipeline(
                    "text-generation",
                    model=model_dir,
                    model_kwargs={"torch_dtype": torch.bfloat16,},
                    device_map="auto",
                    # quantization_config=quantization_config
                )
            else:
                self.pipe, self.tokenizer = load_model(
                    model_dir,
                    'cuda',
                    self.args.num_gpus,
                    dtype=torch.float16,
                    load_8bit=False,
                    cpu_offloading=False,
                    revision=None,
                    debug=False,
                )
            self.system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            # if 'llama-3' in model:
            #     model_dir = 'meta-llama/Meta-L' + model[1:]
            #     self.pipe = pipeline(
            #         "text-generation",
            #         model=model_dir,
            #         model_kwargs={"torch_dtype": torch.bfloat16, "load_in_8bit": True,},
            #         device_map="auto",
            #         # quantization_config=quantization_config
            #     )

            # else:
            
            #     # local_rank, world_size = setup_model_parallel()
            #     # if local_rank > 0:
            #     #     sys.stdout = open(os.devnull, "w")

            #     # self.generator = load(
            #     #     self.args.ckpt_dir, self.args.tokenizer_path, local_rank, world_size, self.args.max_seq_len, self.args.max_batch_size
            #     # )
            #     # ==================
            #     # self.generator = Llama.build(
            #     #     ckpt_dir=self.args.ckpt_dir,
            #     #     tokenizer_path=self.args.tokenizer_path,
            #     #     max_seq_len=self.args.max_seq_len,
            #     #     max_batch_size=self.args.max_batch_size,
            #     # )
            #     model_dir = os.path.join('/p/project/hai_aaac/models/hf_llama',model)
            #     # model_dir = 'meta-llama/L' + model[1:] + '-hf'
            #     self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto")

            #     if 'chat' in model:
            #         # config = AutoConfig.from_pretrained(model_dir)
            #         # config.pretraining_tp = 1
            #         # self.pipe = AutoModelForCausalLM.from_pretrained(
            #         #     model_dir,
            #         #     config=config,
            #         #     torch_dtype=torch.float16,
            #         #     load_in_8bit=True,
            #         #     device_map='auto'
            #         # )
            #         self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=True)
            #         self.pipe = pipeline(
            #             "text-generation",  # LLM task
            #             model=model_dir,
            #             model_kwargs={"torch_dtype": torch.bfloat16, "load_in_8bit": True,},
            #             device_map="auto",
            #         )

            #     else:
            #         self.tokenizer.pad_token = self.tokenizer.eos_token
            #         self.tokenizer.padding_side = "left"

            #         self.pipe = LlamaForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
            #         self.tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)



        elif 'vicuna' in model:
            model_dir = 'lmsys/' + model

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", use_fast=False)
            self.pipe = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)

        elif 'Baichuan' in model:
            trust_remote_code = True
            model_dir = 'baichuan-inc/' + model

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", use_fast=False, trust_remote_code=trust_remote_code)
            self.pipe = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
            self.pipe.generation_config = GenerationConfig.from_pretrained(model_dir)

        elif 'qwen' in model.lower():
            import socket
            if 'juwels' in socket.gethostname():
                model_dir = os.path.join('../models/qwen',model.lower())
            else:
                model_dir = os.path.join('/data/models', model.lower())
            
            if 'vl' in model.lower():
                self.pipe = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_dir, dtype=torch.bfloat16, device_map="auto"
                )
                
                self.tokenizer = AutoProcessor.from_pretrained(model_dir)
                
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                self.pipe = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.system_prompt = "You are a helpful, respectful and honest assistant. Follow the given examples and answer the question."
                        
        # elif 'chatglm' in model:
        #     model_dir = 'THUDM/' + model
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, cache_dir='/p/project/hai_robtime/.cache/huggingface', device_map="auto")
        #     self.pipe = AutoModel.from_pretrained(model_dir, trust_remote_code=True, cache_dir='/p/project/hai_robtime/.cache/huggingface', device_map="auto").half()

        # elif 'Samantha' in model:
        #     model_dir = 'ehartford/' + model

        #     self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", use_fast=False, cache_dir='/p/project/hai_robtime/.cache/huggingface')
        #     self.pipe = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16, cache_dir='/p/project/hai_robtime/.cache/huggingface')

        # elif 'instruct' in model:
        #     model_dir = 'upstage/' + model
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir='/p/project/hai_robtime/.cache/huggingface')
        #     self.pipe = AutoModelForCausalLM.from_pretrained(
        #         model_dir,
        #         device_map="auto",
        #         torch_dtype=torch.float16,
        #         load_in_8bit=True,
        #         rope_scaling={"type": "dynamic", "factor": 2}, # allows handling of longer inputs
        #         cache_dir='/p/project/hai_robtime/.cache/huggingface'
        #     )


    def eval(self, preds, gts):
        
        if self.args.dataset in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli", "mmlu", "adv_sst2", "adv_qqp", "adv_mnli", "adv_mnli-mm", "adv_qnli", "adv_rte"]:
            if self.args.dataset == "mmlu":
                preds = [pred.lower() for pred in preds]
                gts = [gt.lower() for gt in gts]
            
            if not isinstance(preds, list):
                preds = [preds]
                gts = [gts]

            return sum(a == b for a, b in zip(preds, gts)) / len(preds)
        
        elif self.args.dataset == "squad_v2":

            from metrics.squad_v2.squad_v2 import SquadV2
            metric = SquadV2()

            model_output = []

            for id, pred in zip(gts, preds):

                if pred == "unanswerable":
                    no_ans_prob = 1
                    pred = ""
                else:
                    no_ans_prob = 0
                
                model_output.append({"id": id, "prediction_text": pred, "no_answer_probability": no_ans_prob})

            references = self.args.data.get_reference()
            score = metric.compute(predictions=model_output, references=references)

            return score["f1"] / 100
        
        elif self.args.dataset in ['iwslt', 'un_multi']:

            from metrics.bleu.bleu import Bleu
            metric = Bleu()
            results = metric.compute(predictions=preds, references=gts)
            
            # it need to /100 to get the proper bleu score (in alignment with other dataset, e.g., glue) 
            return results['bleu'] / 100

        elif self.args.dataset == 'math':
  
            processed_preds = []
            processed_gts = []
            for pred, gt in zip(preds, gts):
                if pred.lower() == "yes":
                    pred = "True"
                elif pred.lower() == "no":
                    pred = "False"
                
                gt = str(gt).lower()
                processed_preds.append(pred.lower())
                processed_gts.append(gt.lower())

            acc = sum(a == b for a, b in zip(processed_preds, processed_gts)) / len(processed_gts)

            return acc
            
        else:
            raise NotImplementedError("Eval this dataset {self.args.dataset} is not implemented!")
        
    def predict_gpt(self, prompts, prompt_dict):
        prompt_dict = self.predict_by_local_inference_batch_gpt(self.model, prompts, prompt_dict)
        
        return prompt_dict


    def predict(self, prompt=None):
        assert self.args.data is not None, "Please load data first!"
        # elif 'llama' in self.model.lower():
        #     results = self.predict_by_local_inference(self.model, prompt)

        results = self.predict_by_local_inference_hfmodel(self.model, prompt)

        return results
    
    def get_prompt(self, message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)
    
    def get_input_token_length(self, message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
        prompt = self.get_prompt(message, chat_history, system_prompt)
        input_ids = self.tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
        return input_ids.shape[-1]
    
    def generate(self,
        message: str,
        history_with_input: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Iterator[list[tuple[str, str]]]:
        if max_new_tokens > 2048:
            raise ValueError

        history = history_with_input[:-1]
        generator = self.run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
        try:
            first_response = next(generator)
            yield history + [(message, first_response)]
        except StopIteration:
            yield history + [(message, '')]
        for response in generator:
            yield history + [(message, response)]

    def run(self,
            message: str,
            chat_history: list[tuple[str, str]],
            system_prompt: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 50) -> Iterator[str]:
        prompt = self.get_prompt(message, chat_history, system_prompt)
        inputs = self.tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')

        streamer = TextIteratorStreamer(self.tokenizer,
                                        timeout=10.,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=self.pipe.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield ''.join(outputs)
            
    def apply_chat(self, tokenizer, messages, add_generation_prompt=True, enable_thinking=None, **kwargs,):
        sig = inspect.signature(tokenizer.apply_chat_template)
        breakpoint()
        has_enable_thinking = "enable_thinking" in sig.parameters

        base_kwargs = dict(
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if enable_thinking is not None and has_enable_thinking:
            # 只有在函数签名里确实有这个参数时才传
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=enable_thinking,
                **base_kwargs,
            )
        else:
            # 对于没有 enable_thinking 的模型，正常调用
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                **base_kwargs,
            )

    def predict_by_local_inference_hfmodel(self, model, prompt):
        data_len = len(self.args.data)
        if data_len > 500:
            data_len = 500
    
        score = 0
        check_correctness = 100
        preds = []
        gts = []
        
        for idx in tqdm(range(data_len)):
            
            raw_data = self.args.data.get_content_by_idx(idx, self.args.dataset.split("adv_")[-1])
            input_text, gt = self.process_input(prompt, raw_data)
            # if 'llama-3' in model.lower() and 'instruct' not in model.lower():
            if 'llama-3' in model.lower():
                label_ids = [self.pipe.tokenizer.encode(" " + str(label))[-1] for label in LABEL_SET_NEW[self.args.dataset.split("adv_")[-1]]]

                def prefix_allowed_tokens_fn(batch_id, input_ids):
                    return label_ids
                
                if 'instruct' in model.lower():
                    message = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                    ]

                else:
                    message = input_text
                
                # terminators = [
                #     self.pipe.tokenizer.eos_token_id,
                #     self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                # ]
                outputs = self.pipe(
                    message,
                    max_new_tokens=self.args.generate_len,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    # eos_token_id=terminators,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                raw_pred = outputs[0]["generated_text"][len(message):]
                if 'instruct' in model.lower():
                    raw_pred = outputs[0]["generated_text"][-1]['content']
                
            #     # if 'instruct' in model.lower():
            #     messages = [
            #         {"role": "system", "content": "Follow the given examples and answer the question."},
            #         {"role": "user", "content": input_text},
            #     ]
            #     input_text = self.pipe.tokenizer.apply_chat_template(
            #             messages, 
            #             tokenize=False, 
            #             add_generation_prompt=True
            #     )

            #     terminators = [
            #         self.pipe.tokenizer.eos_token_id,
            #         self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            #     ]
            #     outputs = self.pipe(
            #         input_text,
            #         max_new_tokens=self.args.generate_len,
            #         eos_token_id=terminators,
            #         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            #         do_sample=True,
            #         temperature=0.6,
            #         top_p=0.9,
            #     )
            #     raw_pred = outputs[0]["generated_text"][len(input_text):]
            # else:
            #     if 'chat' in model.lower():
            #         if 'llama' in model:
            #             # generator = self.generate(input_text, [], DEFAULT_SYSTEM_PROMPT, 40, 1, 0.95, 50)
            #             # for x in generator:
            #             #     pass
            #             # _, raw_pred = x[0]

            #             label_ids = [self.tokenizer.encode(" " + str(label))[-1] for label in LABEL_SET_NEW[self.args.dataset.split("adv_")[-1]]]

            #             def prefix_allowed_tokens_fn(batch_id, input_ids):
            #                 return label_ids

            #             outputs = self.pipe(
            #                 input_text,
            #                 max_new_tokens=self.args.generate_len,
            #                 eos_token_id=self.tokenizer.eos_token_id,
            #                 prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            #                 do_sample=True,
            #                 temperature=0.6,
            #                 top_p=0.9,
            #             )
            #             raw_pred = outputs[0]["generated_text"][len(input_text):]
            #         elif 'baichuan' in model.lower():
            #             raw_pred = self.pipe.chat(self.tokenizer, [{"role": "user", "content": prompt}])

            #     else:
            #         raw_pred = self.pred_by_generation_hfmodel(input_text, model)
            
            elif 'qwen' in model.lower():
                label_ids = [self.tokenizer.encode(" " + str(label))[-1] for label in LABEL_SET_NEW[self.args.dataset.split("adv_")[-1]]]
                def prefix_allowed_tokens_fn(batch_id, input_ids):
                    return label_ids
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_text},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.pipe.device)

                generated_ids = self.pipe.generate(
                    **model_inputs,
                    pad_token_id=self.tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    max_new_tokens=self.args.generate_len,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                raw_pred = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            else:

                label_ids = [self.tokenizer.encode(" " + str(label))[-1] for label in LABEL_SET_NEW[self.args.dataset.split("adv_")[-1]]]
                def prefix_allowed_tokens_fn(batch_id, input_ids):
                    return label_ids
                
                conv_temp = get_conversation_template(self.args.model)
                if 'instruct' in self.args.model.lower():
                    conv_temp.set_system_message(self.system_prompt)

                conv_temp.append_message(conv_temp.roles[0], input_text)
                conv_temp.append_message(conv_temp.roles[1], None)

                prompt_input = conv_temp.get_prompt()
                input_ids = self.tokenizer([prompt_input]).input_ids
                output_ids = self.pipe.generate(
                    torch.as_tensor(input_ids).cuda(),
                    # do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    max_new_tokens=self.args.generate_len,
                )

                if self.pipe.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(input_ids[0]):]

                raw_pred = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                )
            
            pred = self.process_pred(raw_pred)

            preds.append(pred)
            gts.append(gt)
 
            if check_correctness > 0:
                self.args.logger.info("gt: {}".format(gt))
                self.args.logger.info("Pred: {}".format(pred))
                self.args.logger.info("sentence: {}".format(input_text))

                check_correctness -= 1

        score = self.eval(preds, gts)
        return score

    def pred_by_generation_hfmodel(self, input_text, model):
        # out = 'error!'

        # if 'vicuna' in model:
        #     input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        #     outputs = self.pipe.generate(input_ids, 
        #                                  temperature=0,
        #                                  max_new_tokens=self.args.generate_len, 
        #                                  early_stopping=True)
        #     out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # elif 'llama' in model:
        #     def get_tokens_as_tuple(word):
        #         return tuple(self.tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])
            
        #     if 'chat' in model:
        #         dialogs = [[{"role": "user", "content": input_text}]]

        #         input_ids = self.chat_completion(dialogs)

                

        #     else:
        #     sequence_bias = {}
        #     for word in LABEL_SET[self.args.dataset.split("adv_")[-1]]:
        #         if not isinstance(word, str):
        #             word = str(word)
        #         sequence_bias[get_tokens_as_tuple(word)] = 100.0

        toks = self.tokenizer(input_text, return_tensors="pt")
        
        newline_token_id = self.tokenizer.encode("\n")[-1]
        label_ids = [self.tokenizer.encode(" " + str(label))[-1] for label in LABEL_SET_NEW[self.args.dataset.split("adv_")[-1]]]

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            if 'llama' in model or 'qwen' in model.lower():
                return label_ids
            else:
                pass

        # def get_tokens_as_tuple(word):
        #     return tuple(self.tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])
        
        # sequence_bias = {}
        # for word in LABEL_SET[self.args.dataset.split("adv_")[-1]]:
        #     if not isinstance(word, str):
        #         word = str(word)
        #     sequence_bias[get_tokens_as_tuple(word)] = 100.0

        if 'glm' in model:
            response, history = self.pipe.chat(self.tokenizer, input_text, history=[])
            
        elif 'llama' in model:
            if 'instruct' in model:
                del toks["token_type_ids"]
                toks = toks.to("cuda")
                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

                output = self.pipe.generate(**toks, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            else:
                outputs = self.pipe.generate(toks["input_ids"].to("cuda"),
                                            attention_mask=toks["attention_mask"].to("cuda"),
                                            max_new_tokens=self.args.generate_len,
                                            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                            eos_token_id=newline_token_id,
                                            # sequence_bias=sequence_bias,           
                                            do_sample=True,
                                            num_beams=4,
                                            )
                
                decoded_outputs = self.tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                full_response = decoded_outputs.lstrip(self.tokenizer.pad_token)

                try:
                    # If multiple extra lines were printed, only take the first one
                    response = full_response[len(input_text):].split("\n")[0]
                
                except:
                    response = full_response
                
                # Clean up any extra spaces at the start, if necessary
                if len(response) == 0:
                    response = full_response

                if response[0] == " ":
                    if len(response) == 1:
                        response = full_response
                    else:
                        response = response[1:]
        
        elif 'qwen' in model.lower():        
            outputs = self.pipe.generate(toks["input_ids"].to("cuda"),
                                            attention_mask=toks["attention_mask"].to("cuda"),
                                            max_new_tokens=self.args.generate_len,
                                            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                            eos_token_id=newline_token_id,
                                            # sequence_bias=sequence_bias,           
                                            do_sample=True,
                                            num_beams=4,
                                            )
                
            decoded_outputs = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            full_response = decoded_outputs.lstrip(self.tokenizer.pad_token)
        
        return response

        # out = 'error!'

        # if model in ["llama-7b", "llama-13b", "llama-30b", "llama-65b"]:
        #     results = self.generator.generate(
        #         [input_text],
        #         max_gen_len=self.args.generate_len,
        #         temperature=self.args.temperature,
        #         top_p=self.args.top_p,
        #     )

        #     out = results[0]
            
        # elif model in ["llama-2-7b", "llama-2-13b", "llama-2-70b"]:
        #     results = self.generator.text_completion(
        #         [input_text],
        #         max_gen_len=self.args.generate_len,
        #         temperature=self.args.temperature,
        #         top_p=self.args.top_p,
        #     )

        #     out = results[0]['generation']

        # elif model in ["llama-2-7b-chat", 'llama-2-13b-chat', 'llama-2-70b-chat']:
        #     dialogs = [[{"role": "user", "content": input_text}]]

        #     results = self.generator.chat_completion(
        #         dialogs,  # type: ignore
        #         max_gen_len=self.args.generate_len,
        #         temperature=self.args.temperature,
        #         top_p=self.args.top_p,
        #     )

        #     out = results[0]['generation']['content']

        # return out

    def chat_completion(
        self,
        dialogs: List[Dialog],
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """

        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        return_tensors="pt",
                    ).input_ids.to("cuda")
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            tmp = self.tokenizer(f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}", return_tensors="pt").input_ids.to("cuda")

            dialog_tokens += tmp
            prompt_tokens.append(dialog_tokens)

        return prompt_tokens[0][0].unsqueeze(0)
    
    def build_batch_bytes_for_requests(self, requests):
        """
        requests: list of dict with keys:
        - custom_id
        - messages
        - model (judge model)
        - generation params
        return: BytesIO (file-like)
        """
        buf = io.BytesIO()
        for req in requests:
            line = {
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req["model"],
                    "messages": req["messages"],
                    "temperature": req.get("temperature", 0.1),
                    "max_completion_tokens": req.get("max_completion_tokens", 256),
                    "top_p": req.get("top_p", 0.9),
                    "frequency_penalty": req.get("frequency_penalty", 0),
                    "presence_penalty": req.get("presence_penalty", 0),
                },
            }
            buf.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
        buf.seek(0)
        return buf
    
    def wait_for_batch_and_get_predictions(self, client, batch_id, poll_interval=30):
        terminal_failed_states = {"failed", "expired", "canceled"}
        while True:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            print(f"[batch {batch_id}] status={status}")

            if status == "completed":
                content = client.files.content(batch.output_file_id).read()
                preds = {}  # custom_id -> assistant content
                for line in content.decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    cid = obj.get("custom_id")
                    assistant_content = None
                    resp = obj.get("response") or {}
                    body = resp.get("body") or {}
                    for choice in body.get("choices", []):
                        msg = choice.get("message", {})
                        if msg.get("role") == "assistant":
                            assistant_content = msg.get("content")
                            break
                    preds[cid] = assistant_content
                return preds

            if status in terminal_failed_states:
                print(f"[batch {batch_id}] FAILED: {status}")
                if getattr(batch, "errors", None):
                    for e in batch.errors:
                        print(e)
                raise RuntimeError(f"Batch failed: {status}")

            time.sleep(poll_interval)
    
    def predict_by_local_inference_batch_gpt(self, model, prompts, prompt_dict):
        requests = []
        gts = {}
        
        data_len = len(self.args.data)
        if data_len > 500:
            data_len = 500
        
        client = OpenAI()
        for i, prompt in enumerate(prompts):
            gts[prompt] = []
            for j in range(data_len):
                raw_data = self.args.data.get_content_by_idx(j, self.args.dataset.split("adv_")[-1])
                input_text, gt = self.process_input(prompt, raw_data)
                cid = f"{model}-{self.args.dataset}-{i}-{j}"  # 为保证唯一性（否则 batch 会报错）

                messages = [
                    {"role": "system", "content": "Follow the given examples and answer the question."},
                    {"role": "user", "content": input_text},
                ]
                requests.append({
                    "custom_id": cid,
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": self.args.generate_len,
                })
                
                gts[prompt].append(gt)
            
        buf = self.build_batch_bytes_for_requests(requests)
        file_obj = client.files.create(file=buf, purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        raw_preds = self.wait_for_batch_and_get_predictions(client, batch.id)
        
        # batch_id = 'batch_6941d6db9b50819090f0fa6a4fe07ecf'
        # batch = client.batches.retrieve(batch_id)
        # status = batch.status
        # print(f"[batch {batch_id}] status={status}")

        # content = client.files.content(batch.output_file_id).read()
        # raw_preds = {}  # custom_id -> assistant content
        # for line in content.decode("utf-8").splitlines():
        #     if not line.strip():
        #         continue
        #     obj = json.loads(line)
        #     cid = obj.get("custom_id")
        #     assistant_content = None
        #     resp = obj.get("response") or {}
        #     body = resp.get("body") or {}
        #     for choice in body.get("choices", []):
        #         msg = choice.get("message", {})
        #         if msg.get("role") == "assistant":
        #             assistant_content = msg.get("content")
        #             break
        #     raw_preds[cid] = assistant_content
        
        for idx, prompt in enumerate(prompts):
            preds = []
            gtsss = []
            for cid, pred in raw_preds.items():
                if idx != int(cid.split('-')[-2]):
                    continue
                pred = self.process_pred(pred)
                gt = gts[prompt][int(cid.split('-')[-1])]
                self.args.logger.info("gt: {}".format(gt))
                self.args.logger.info("Pred: {}".format(pred))
                self.args.logger.info("sentence: {}".format(input_text))
            
                preds.append(pred)
                gtsss.append(gt)
            
            acc = self.eval(preds, gtsss)
            prompt_dict[prompt] = acc
            self.args.logger.info("{:.3f}%, {}\n".format(acc*100, prompt))
            self.args.logger.info("=================================================")
   
        return prompt_dict

    def predict_by_local_inference_chatgpt(self, model, prompt):
        client = OpenAI()

        data_len = len(self.args.data)
        if data_len > 500:
            data_len = 500
    
        score = 0
        check_correctness = 100
        preds = []
        gts = []
        idx = 0
        # for idx in tqdm(range(data_len)):
        pbar = tqdm(total=data_len)
        while idx < data_len:
            raw_data = self.args.data.get_content_by_idx(idx, self.args.dataset.split("adv_")[-1])
            input_text, gt = self.process_input(prompt, raw_data)
            
            try:
                completion = client.chat.completions.create(
                    # model="gpt-3.5-turbo",
                    model=model,
                    messages=[
                        {"role": "system", "content": "Follow the given examples and answer the question."},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0,
                    max_tokens=self.args.generate_len
                )
                raw_pred = completion.choices[0].message.content
                
            except:
                import time
                time.sleep(100)
                continue

            # raw_pred = self.pred_by_generation(input_text, model)
            pred = self.process_pred(raw_pred)

            preds.append(pred)
            gts.append(gt)
 
            if check_correctness > 0:
                self.args.logger.info("gt: {}".format(gt))
                self.args.logger.info("Pred: {}".format(pred))
                self.args.logger.info("sentence: {}".format(input_text))

                check_correctness -= 1

            idx += 1
            pbar.update(1)
        pbar.close()

        score = self.eval(preds, gts)
        return score


    def _process_math_input(self, prompt, raw_data):
        from config import MATH_QUESTION_TYPES
        question_type, question, label = MATH_QUESTION_TYPES[raw_data['task']], raw_data['question'], raw_data['answer']
        input_text = prompt.format(question_type) + '\n'
        
        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(raw_data['task'])
        
        input_text += ("Question: " + question + '\nAnswer: ')

        return input_text, label


    def _process_trans_input(self, prompt, raw_data):
        from config import LANGUAGES
        source, target, task = raw_data['source'], raw_data['target'], raw_data['task']
        src_lang, des_lang = task.split('-')
        input_text = prompt.format(LANGUAGES[src_lang], LANGUAGES[des_lang]) + '\n'

        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(task)

        input_text += (source + '\nAnswer: ')
        return input_text, target


    def _process_squad_v2_input(self, prompt, raw_data):
        id, content = raw_data["id"], raw_data["content"]
        input_text = prompt

        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(self.args.dataset)

        input_text += (content + "Answer: ")

        return input_text, id


    def _process_qa_input(self, prompt, raw_data):
        task, content = raw_data["task"], raw_data["content"]
        label = raw_data["label"]

        input_text = prompt.format(task) + "\n"

        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(task.replace(" ", "_"))
        
        input_text += content + "\nAnswer: "
        
        return input_text, label


    def _process_cls_input(self, prompt, raw_data):
        content = raw_data["content"]
        label = raw_data["label"]

        input_text = prompt

        if self.args.shot > 0:
            few_shot_examples = self.args.data.get_few_shot_examples(self.args.dataset)
            input_text += "\n"+few_shot_examples
            if  self.args.dataset == "sst2" or  self.args.dataset == "cola":
                input_text += "Sentence: "
        suffix = ''
        # if 'llama' in self.args.model:
        #     suffix = ANSWER_WORD[self.args.dataset]
        input_text += (content + ' Answer: ' + suffix)

        return input_text, label

    
    def _process_math_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(r",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred
    

    def _process_trans_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(r",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred


    def _process_squad_v2_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(r",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred


    def _process_cls_pred(self, raw_pred):

        pred = raw_pred.lower()

        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.replace("'", "")
        pred = pred.replace("\"", "")

        # pred = pred.strip(r",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        # pred = pred.split(" ")[-1]

        # pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        # if pred in LABEL_SET[self.args.dataset.split("adv_")[-1]]:                   
        #     pred = LABEL_TO_ID[self.args.dataset.split("adv_")[-1]][pred] 

        new_pred = -1
        for word in LABEL_SET_NEW[self.args.dataset.split("adv_")[-1]]:
            if word.lower() in pred:
                new_pred = LABEL_TO_ID[self.args.dataset.split("adv_")[-1]][word]

        # import re
        # if len(re.findall('entailment', pred)) > 1:
        #     new_pred = -1
            
            
        if new_pred == -1:
            self.args.logger.warn("The original label : '{}'.".format(raw_pred))                
            self.args.logger.warn("The predicted label: '{}' is not in label set.".format(pred))
        
        return new_pred
    
    
    def _process_qa_pred(self, raw_pred):
        pred = raw_pred.lower()
        
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        
        pred = pred.strip(r",._\"\'-+¬=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
        pred = pred.split(" ")[-1]
        pred = pred.strip(r",._\"\'-+¬=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
        
        if pred not in LABEL_SET[self.args.dataset.split("adv_")[-1]]:   
            self.args.logger.warn("The original label : '{}'.".format(raw_pred))
            self.args.logger.warn("The predicted label: '{}' is not in label set.".format(pred))
            pred = 'no_answer'

        return pred
    

def setup_model_parallel() -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(rank)

    # seed must be the same in all processes
    # return local_rank, world_size
    return rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def load_conversation_template(template_name):
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template