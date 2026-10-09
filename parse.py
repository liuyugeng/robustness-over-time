import numpy as np
import re


def split_markdown_by_title(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()

    re_str = "# cola|# mnli|# mrpc|# qnli|# qqp|# rte|# sst2|# wnli|# mmlu|# squad_v2|# iwslt|# un_multi|# math"
    
    datasets = ["# cola", "# mnli", "# mrpc", "# qnli", "# qqp", "# rte", "# sst2", "# wnli",
                "# mmlu", "# squad_v2", "# iwslt", "# un_multi", "# math"]
    
    # re_str = "# cola|# mnli|# mrpc|# qnli|# qqp|# rte|# sst2|# wnli"
    # datasets = ["# cola", "# mnli", "# mrpc", "# qnli", "# qqp", "# rte", "# sst2", "# wnli"]
    primary_sections = re.split(re_str, content)[1:]
    assert len(primary_sections) == len(datasets)

    all_sections_dict = {}

    for dataset, primary_section in zip(datasets, primary_sections):
        re_str = "## "
        results = re.split(re_str, primary_section)
        keywords = ["10 prompts", "bertattack", "checklist", "deepwordbug", "stresstest",
                    "textfooler", "textbugger", "translation"]

        secondary_sections_dict = {}
        for res in results:
            for keyword in keywords:
                if keyword in res.lower():
                    secondary_sections_dict[keyword] = res
                    break

        all_sections_dict[dataset] = secondary_sections_dict

    return all_sections_dict
# def prompts_understanding(sections_dict):
#     for dataset in sections_dict.keys():
#         # print(dataset)
#         for title in sections_dict[dataset].keys():
#             if title == "10 prompts":
#                 prompts = sections_dict[dataset][title].split("\n")
#                 num = 0
#                 task_prompts_acc = []
#                 role_prompts_acc = []
#                 for prompt in prompts:
#                     if "Acc: " not in prompt:
#                         continue
#                     else:
#                         import re
#                         num += 1
#                         match = re.search(r'Acc: (\d+\.\d+)%', prompt)
#                         if match:
#                             number = float(match.group(1))
#                         if num <= 10:
#                             task_prompts_acc.append(number)
#                         else:
#                             role_prompts_acc.append(number)

#                 print(task_prompts_acc)
#                 print(role_prompts_acc)
import os
def list_files(directory):
    files = [os.path.join(directory, d) for d in os.listdir(directory) if not os.path.isdir(os.path.join(directory, d))]
    return files

def convert_model_name(attack):
    attack_name = {
        "t5": "t5",
        "UL2": "ul2",
        "Vicuna": "vicuna",
        "chatgpt": "chatgpt",
    }
    return attack_name[attack]

def convert_attack_name(attack):
    attack_name = {
        "BertAttack".lower(): "bertattack",
        "CheckList".lower(): "checklist",
        "DeepWordBug".lower(): "deepwordbug",
        "StressTest".lower(): "stresstest",
        "TextFooler".lower(): "textfooler",
        "TextBugger".lower(): "textbugger",
        "Semantic".lower(): "translation",
    }
    return attack_name[attack]

def convert_dataset_name(dataset):
    dataset_name = {
        "CoLA".lower(): "# cola",
        "MNLI".lower(): "# mnli",
        "MRPC".lower(): "# mrpc",
        "QNLI".lower(): "# qnli",
        "QQP".lower(): "# qqp",
        "RTE".lower(): "# rte",
        "SST2".lower(): "# sst2",
        "WNLI".lower(): "# wnli",
        "MMLU".lower(): "# mmlu",
        "SQuAD_V2".lower(): "# squad_v2",
        "IWSLT".lower(): "# iwslt",
        "UN_Multi".lower(): "# un_multi",
        "Math".lower(): "# math",
        "Avg".lower(): "Avg",
    }
    return dataset_name[dataset]


def retrieve(model_name, dataset_name, attack_name, prompt_type):
    # model_name = convert_model_name(model_name)
    dataset_name = convert_dataset_name(dataset_name)
    attack_name = convert_attack_name(attack_name)

    if "zero" in prompt_type:
        shot = "zeroshot"
    else:
        shot = "fewshot"
    
    # if "task" in prompt_type:
    #     prompt_type = "task"
    # else:
    #     prompt_type = "role"
    
    directory_path = "./"
    md_dir = os.path.join(directory_path, model_name.lower() + "_" + shot + ".md")
    sections_dict = split_markdown_by_title(md_dir)
    results = {}
    for cur_dataset in sections_dict.keys():
        if cur_dataset == dataset_name:
            dataset_dict = sections_dict[cur_dataset]
            best_acc = 0
            best_prompt = ""
            for cur_attack in dataset_dict.keys():
                if cur_attack == "10 prompts":
                    prompts_dict = dataset_dict[cur_attack].split("\n")
                    num = 0
                    for prompt_summary in prompts_dict:
                        if "Acc: " not in prompt_summary:
                            continue
                        else:
                            import re
                            num += 1
                            match = re.search(r'Acc: (\d+\.\d+)%', prompt_summary)
                            if match:
                                number = float(match.group(1))
                                if number > best_acc:
                                    best_acc = number
                                    best_prompt = prompt_summary.split("prompt: ")[1]

            for cur_attack in dataset_dict.keys():
                
                if cur_attack == attack_name:

                    if attack_name == "translation":
                        prompts_dict = dataset_dict[attack_name].split("\n")
            
                        for prompt_summary in prompts_dict:
                            if "acc: " not in prompt_summary:
                                continue
                            
                            prompt = prompt_summary.split("prompt: ")[1]
                            
                            import re
                        
                            match_atk = re.search(r'acc: (\d+\.\d+)%', prompt_summary)
                            number_atk = float(match_atk.group(1))
                            results[prompt] = number_atk
  
                        sorted_results = sorted(results.items(), key=lambda item: item[1])

                        returned_results = []
                        for result in sorted_results:
                            returned_results.append({"origin prompt": best_prompt, "origin acc": best_acc, "attack prompt": result[0], "attack acc": result[1], "dropped acc": best_acc-result[1]})
      
                        return returned_results

                    elif attack_name in ["bertattack", "checklist", "deepwordbug", "stresstest", "textfooler", "textbugger"]:

                        prompts_dict = dataset_dict[attack_name].split("Original prompt: ")
                        num = 0
                        
                        returned_results = []
                        for prompt_summary in prompts_dict:
                            if "Attacked prompt: " not in prompt_summary:
                                continue

                            origin_prompt = prompt_summary.split("\n")[0]
                            attack_prompt = prompt_summary.split("Attacked prompt: ")[1].split("Original acc: ")[0]
                            attack_prompt = bytes(attack_prompt[2:-1], "utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")

                            # print(origin_prompt)
                            # print(attack_prompt)

                            num += 1
                            import re
                            match_origin = re.search(r'Original acc: (\d+\.\d+)%', prompt_summary)
                            match_atk = re.search(r'attacked acc: (\d+\.\d+)%', prompt_summary)
                            match_drp = re.search(r'dropped acc: (\d+\.\d+)%', prompt_summary)
                            if match_origin and match_atk:
                                # if prompt_type == "task":
                                #     if num > 3:
                                #         break
                                # else:
                                #     if num < 3:
                                #         continue
                                number_origin = float(match_origin.group(1))
                                number_atk = float(match_atk.group(1))
                                number_drp = number_origin - number_atk # float(match_drp.group(1))
                                returned_results.append({"origin prompt": origin_prompt, "origin acc": number_origin, "attack prompt": attack_prompt, "attack acc": number_atk, "dropped acc": number_drp})
                        
                        return returned_results