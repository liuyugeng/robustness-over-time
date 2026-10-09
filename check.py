import re
import os
import sys
import glob
import pickle
import argparse

from config import *
from tqdm import tqdm
from datetime import datetime

def check_gpt(args):
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
        
    LOGS_DIR = os.path.join("./logs", args.dataset + '/')
    file_name = args.model.replace('/', '_') + '_' + args.attack + "_gen_len_" + str(args.generate_len) + "_" + str(args.shot) + "_shot_" + args.arch + "_dem_" + str(args.clean)
    pattern = re.compile(rf"^{file_name}_(\d{{4}}-\d{{2}}-\d{{2}})\.log$")

    files = []

    # 找匹配的文件
    for name in os.listdir(LOGS_DIR):
        m = pattern.match(name)
        if m:
            date_str = m.group(1)
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                files.append((dt, name))
            except ValueError:
                pass

    files.sort()  # 日期升序：最早的在前

    # 情况 1：如果恰好有 1 个文件，退出程序
    if len(files) == 1:
        # print(f"文件已存在：{files[0][1]}，程序退出。")
        sys.exit(0)

    # 情况 2：如果超过 1 个，删除日期靠前的（最早的）
    elif len(files) > 1:
        to_delete = files[0][1]
        os.remove(os.path.join(LOGS_DIR, to_delete))
        # print(f"检测到超过一个文件，已删除最早的：{to_delete}")
        sys.exit(0)
        
    else:
        print(file_name)


def rewrite():
    path = os.path.join("../logs", "output*.log")


    for file in sorted(glob.glob(path)):
        # number = int(file.split('-')[-1].split('.')[0])
        number = int(re.findall(r"-(.+?).log", file)[0])
        if number > 8432074:
            with open(file, "r") as f1, open(file+"bak", "w") as f2:
                content = f1.readlines()
                for c in content:
                    if 'done' in c:
                        new_c = c.split("  ")[0] + ' 70b-chat ' + c.split("  ")[1]
                        f2.write(new_c)

                    else:
                        f2.write(c)
            os.remove(file)
            os.rename(file+"bak", file)


def save():
    path = os.path.join("../logs", "output*.log")

    finished = {}

    for file in sorted(glob.glob(path)):
        # number = int(file.split('-')[-1].split('.')[0])
        number = int(re.findall(r"-(.+?).log", file)[0])
        if number < 8415755:
            continue
       
        with open(file, 'r') as r:
            content = r.read()

            dataset = re.findall(r"'dataset': '(.+?)'", content)
            if len(dataset) == 0:
                continue

            shot = re.findall(r"'shot': (.+?)", content)[0]
            clean = re.findall(r"'clean': (.+?)", content)[0]
            
            mat = re.findall(r'(.+?) done', content)
            for i in range(len(mat)):
                if shot == '3':
                    mat[i] += ' ' + shot + ' ' + clean
                else:
                    mat[i] += ' ' + shot 
            if dataset[0] in finished.keys():
                finished[dataset[0]] += mat
            else:
                finished[dataset[0]] = mat

            # print(mat)

    with open('finished.pkl', 'wb') as f:  
        pickle.dump(finished, f)


def check_2(args):
    with open('finished.pkl', 'rb') as f:  
        finished = pickle.load(f)

    if args.model == 'llama-2-7b-chat' or args.model == 'llama-2-13b-chat':
        string = f"{args.arch} {args.dataset} {args.attack} {str(args.shot)}" 
        if args.shot == 3:
            string += f" {str(args.clean)}"
        if string in finished[args.dataset]:
            return True
        else:
            return False
        
    elif "llama-2" in args.model and args.model != 'llama-2-7b-chat' and args.model != 'llama-2-13b-chat':
        model = args.model.split("llama-2-")[-1]
        string = f"{args.arch} {model} {args.dataset} {args.attack} {str(args.shot)}" 
        if args.shot == 3:
            string += f" {str(args.clean)}"
        if string in finished[args.dataset]:
            return True
        else:
            return False
        
    else:
        model = args.model.split("llama-")[-1]
        string = f"{args.arch} {model} {args.dataset} {args.attack} {str(args.shot)}" 
        if args.shot == 3:
            string += f" {str(args.clean)}"
        if string in finished[args.dataset]:
            return True
        else:
            return False
        
def check(args):
    with open('finished.pkl', 'rb') as f:  
        finished = pickle.load(f)

    if args.model == 'llama-2-7b-chat' or args.model == 'llama-2-13b-chat':
        string = f"{args.arch} {args.dataset} {args.attack} {str(args.shot)}" 
        if args.shot == 3:
            if args.clean:
                string += " T"
            else:
                string += " F"
        if string in finished[args.dataset]:
            return True
        else:
            return False
        
    elif "llama-2" in args.model and args.model != 'llama-2-7b-chat' and args.model != 'llama-2-13b-chat':
        model = args.model.split("llama-2-")[-1]
        string = f"{args.arch} {model} {args.dataset} {args.attack} {str(args.shot)}" 
        if args.shot == 3:
            if args.clean:
                string += " T"
            else:
                string += " F"
        if string in finished[args.dataset]:
            return True
        else:
            return False
        
    else:
        model = args.model.split("llama-")[-1]
        string = f"{args.arch} {model} {args.dataset} {args.attack} {str(args.shot)}" 
        if args.shot == 3:
            if args.clean:
                string += " T"
            else:
                string += " F"
        if string in finished[args.dataset]:
            return True
        else:
            return False

def rest(args):
    unfinished = {}

    arch = ["t5", "ul2", "vicuna", "chatgpt"]
    attack=['textfooler', 'textbugger', 'deepwordbug', 'checklist', 'stresstest', 'semantic']
    dataset_clean = ["sst2", "cola", "qqp", "mnli", "qnli", "wnli", "rte", "mrpc", "mmlu", "squad_v2", "adv_sst2", "adv_qqp", "adv_mnli", "adv_qnli", "adv_rte"]
    dataset_not = ["sst2", "qqp", "mnli", "qnli","rte", "adv_sst2", "adv_qqp", "adv_mnli", "adv_qnli", "adv_rte"]
    model = ['llama-2-70b-chat', 'llama-2-70b']

    for a in arch:
        for at in attack:
            for d in dataset_clean:
                for m in model:
                    args.arch = a
                    args.attack = at
                    args.dataset = d
                    args.model = m
                    args.shot = 0
                    args.clean = False

                    shot = str(args.shot)
                    clean = str(args.clean)

                    if not check(args):
                        if d in unfinished.keys():
                            unfinished[d].append(a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean)
                        else:
                            unfinished[d] = [a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean]

    for a in arch:
        for at in attack:
            for d in dataset_clean:
                for m in model:
                    args.arch = a
                    args.attack = at
                    args.dataset = d
                    args.model = m
                    args.shot = 3
                    args.clean = True

                    shot = str(args.shot)
                    clean = str(args.clean)

                    if not check(args):
                        if d in unfinished.keys():
                            unfinished[d].append(a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean)
                        else:
                            unfinished[d] = [a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean]

    for a in arch:
        for at in attack:
            for d in dataset_not:
                for m in model:
                    args.arch = a
                    args.attack = at
                    args.dataset = d
                    args.model = m
                    args.shot = 3
                    args.clean = False

                    shot = str(args.shot)
                    clean = str(args.clean)

                    if not check(args):
                        if d in unfinished.keys():
                            unfinished[d].append(a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean)
                        else:
                            unfinished[d] = [a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean]
 


    # path = os.path.join("../logs", "output*.log")

    # for file in sorted(glob.glob(path)):
    #     # number = int(file.split('-')[-1].split('.')[0])
    #     number = int(re.findall(r"-(.+?).log", file)[0])
    #     if number < 8426476:
    #         continue

    #     with open(file, 'r') as r:
    #         content = r.read()
    #         dataset = re.findall(r"'dataset': '(.+?)'", content)
    #         model = re.findall(r"'model': '(.+?)'", content)
    #         if len(model) == 0:
    #             continue
    #         model = model[0]
    #         mat = re.findall(r'(.+?) done', content)
    #         shot = re.findall(r"'shot': (.+?)", content)[0]
    #         arch = ["t5", "ul2", "vicuna", "chatgpt"]
    #         attack=['textfooler', 'textbugger', 'deepwordbug', 'checklist', 'stresstest', 'semantic']
    #         dataset_clean = ["sst2", "cola", "qqp", "mnli", "qnli", "wnli", "rte", "mrpc", "mmlu", "squad_v2", "adv_sst2", "adv_qqp", "adv_mnli", "adv_qnli", "adv_rte"]
    #         dataset_not = ["sst2", "qqp", "mnli", "qnli","rte", "adv_sst2", "adv_qqp", "adv_mnli", "adv_qnli", "adv_rte"]
    #         clean = re.findall(r"'clean': (.+?)", content)[0]
    #         d = dataset[0]
    #         if "llama-2-7b-chat" in model:
    #             if clean == "T":
    #                 for a in arch:
    #                     for k in attack:
    #                         if  a + ' ' + d + ' ' + k not in mat:
    #                             if dataset[0] in unfinished.keys():
    #                                 unfinished[dataset[0]].append(a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean)
    #                             else:
    #                                 unfinished[dataset[0]] = [a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean]
    #             else:
    #                 for a in arch:
    #                     for k in attack:
    #                         if  a + ' ' + d + ' ' + k not in mat:
    #                             if dataset[0] in unfinished.keys():
    #                                 unfinished[dataset[0]].append(a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean)
                                    
    #                             else:
    #                                 unfinished[dataset[0]] = [a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean]

    #         elif "llama-2" in model and model != 'llama-2-7b-chat' and model != 'llama-2-13b-chat':
    #             if clean == "T":
    #                 for a in arch:
    #                     for k in attack:
    #                         if  a + ' ' + model.split("llama-2-")[-1] + ' ' + d + ' ' + k not in mat:
    #                             if dataset[0] in unfinished.keys():
    #                                 unfinished[dataset[0]].append(a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean)
    #                             else:
    #                                 unfinished[dataset[0]] = [a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean]
    #             else:
    #                 for a in arch:
    #                     for k in attack:
    #                         if  a + ' ' + model.split("llama-2-")[-1] + ' ' + d + ' ' + k not in mat:
    #                             if dataset[0] in unfinished.keys():
    #                                 unfinished[dataset[0]].append(a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean)
    #                             else:
    #                                 unfinished[dataset[0]] = [a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean]

    #         else:
    #             if clean == "T":
    #                 for a in arch:
    #                     for k in attack:
    #                         if  a + ' ' + model.split("llama-")[-1] + ' ' + d + ' ' + k not in mat:
    #                             if dataset[0] in unfinished.keys():
    #                                 unfinished[dataset[0]].append(a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean)
    #                             else:
    #                                 unfinished[dataset[0]] = [a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean]
    #             else:
    #                 for a in arch:
    #                     for k in attack:
    #                         if  a + ' ' + model.split("llama-")[-1] + ' ' + d + ' ' + k not in mat:
    #                             if dataset[0] in unfinished.keys():
    #                                 unfinished[dataset[0]].append(a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean)
    #                             else:
    #                                 unfinished[dataset[0]] = [a + ' ' + model + ' ' + d + ' ' + k + ' ' + shot + ' ' + clean]


    for k in unfinished.keys():
        print(k, len(unfinished[k]))

    print(unfinished)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--newmodel', type=str, default='gpt-3.5-turbo')
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

def check_string_in_file(args, target):
    files = glob.glob("./logs/*/" + args.model + "*")
    for file in tqdm(files):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            res = target in content
            if not res:
                print(f"[Error] File not contianed the word: {file}")
                
def normalize_filename(filename, model_prefix):
    """
    将文件名映射为“比较用的 key”，去掉：
    1. 扩展名
    2. 末尾日期（_YYYY-MM-DD）
    3. 开头的 model_prefix + '_'
    
    最终只保留：
        attack_gen_len_..._shot_..._dem_clean
    这样不同 model 的同一条记录可以对齐比较。
    """
    name, _ = os.path.splitext(filename)

    # 去掉末尾的 _YYYY-MM-DD
    name = re.sub(r'_\d{4}-\d{2}-\d{2}$', '', name)

    # 如果是以 model_prefix 开头，去掉 "model_prefix_"
    prefix = model_prefix.replace('/', '_')
    if name.startswith(prefix + "_"):
        name = name[len(prefix) + 1:]

    return name


def get_model_keys_in_dataset(dataset_dir, model_name):
    """
    在某个 dataset 目录下，找到属于某个 model 的所有文件，
    并返回映射后的 key 集合（忽略 model + 日期）。
    """
    model_prefix = model_name.replace('/', '_')
    keys = set()

    # 如果只在 dataset 目录的“当前层”有文件，用 os.listdir 即可：
    for fname in os.listdir(dataset_dir):
        if not os.path.isfile(os.path.join(dataset_dir, fname)):
            continue
        if not fname.startswith(model_prefix):
            continue
        key = normalize_filename(fname, model_prefix)
        keys.add(key)

    # 如果你的文件可能在更深子目录，可以改成 os.walk，例如：
    # for root, _, files in os.walk(dataset_dir):
    #     for fname in files:
    #         if fname.startswith(model_prefix):
    #             full_path = os.path.join(root, fname)
    #             key = normalize_filename(fname, model_prefix)
    #             keys.add(key)

    return keys

def compare_two_models_across_datasets(model1, model2, log_root='./logs'):
    """
    遍历 logs 下所有 dataset 子目录，对比两个 model 的文件覆盖情况。
    返回：
        result[dataset] = (only_in_model1, only_in_model2, in_both)
    """
    res = {}

    # 所有 dataset 子目录
    for dataset in tqdm(sorted(os.listdir(log_root))):
        dataset_dir = os.path.join(log_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        keys1 = get_model_keys_in_dataset(dataset_dir, model1)
        keys2 = get_model_keys_in_dataset(dataset_dir, model2)

        only_1 = keys1 - keys2
        only_2 = keys2 - keys1
        both  = keys1 & keys2

        res[dataset] = (only_1, only_2, both)

    return res

def compare_models(model1, model2):
    result = compare_two_models_across_datasets(model1, model2)

    for dataset, (only_1, only_2, both) in result.items():
        # 如果这一 dataset 下完全没有这两个 model 的文件，可以跳过
        if not (only_1 or only_2 or both):
            continue

        print(f"\n================ Dataset: {dataset} ================")

        if only_1:
            print(f"\n🔴 只在 {model1} 中存在的 key（忽略 model 和日期）:")
            for k in sorted(only_1):
                print("  ", k)

        if only_2:
            print(f"\n🔵 只在 {model2} 中存在的 key（忽略 model 和日期）:")
            for k in sorted(only_2):
                print("  ", k)

        # if both:
        #     print(f"\n✅ 两个 model 都有的 key（忽略 model 和日期）:")
        #     for k in sorted(both):
        #         print("  ", k)
        
if __name__ == '__main__':
    args = get_args()
    # print(check(args))
    # 
    # save()
    # rest(args)
    check_string_in_file(args, 'dropped acc')
    # compare_models(args.model, args.newmodel)


