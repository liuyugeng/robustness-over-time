import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

def compare_last(a, b):
    if (float(a) >= float(b)):
        return 1
    elif abs(float(a) - float(b)) < 0:
        return 1
    else:
        return 0

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='./logs')

    args = parser.parse_args()
    return args

def get_dropped_acc(filepath):
    path = os.path.join(filepath, "*.log")
    result_0 = np.zeros((7, 2))
    result_3 = np.zeros((7, 2))
    model_list = ['chatgpt', '0613']
    attack_list = ['textfooler', 'textbugger', 'bertattack', 'deepwordbug', 'checklist', 'stresstest', 'semantic',]
    for file in glob.glob(path):
        with open(file, 'r') as r:
            content = r.read()
            mat = re.findall(r'dropped acc (.*)%', content)
            # ori_acc = float(mat[0])
            # atk_acc = float(mat[1])
            drp_acc = float(mat[0])
        
        model, attack, shot = file.split("_")[-7].split('/')[-1], file.split("_")[-6], file.split("_")[-2]
        if shot == '0':
            result_0[attack_list.index(attack), model_list.index(model)] = drp_acc
        else:
            result_3[attack_list.index(attack), model_list.index(model)] = drp_acc

    result_0 = pd.DataFrame(result_0, index=attack_list, columns=['0301', '0613'])
    result_3 = pd.DataFrame(result_3, index=attack_list, columns=['0301', '0613'])

    result_0.to_csv(os.path.join(filepath, "0.csv"))
    result_3.to_csv(os.path.join(filepath, "3.csv"))

def read_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    return df

def get_number(path):
    result_0 = read_csv(os.path.join(path, '0.csv'))
    result_0['bool'] = result_0.apply(lambda x: compare_last(x[0],x[1]), axis=1)
    result_3 = read_csv(os.path.join(path, '3.csv'))
    result_3['bool'] = result_3.apply(lambda x: compare_last(x[0],x[1]), axis=1)

    return result_0[result_0['bool'] == 0].shape[0] + result_3[result_3['bool'] == 0].shape[0], result_0[result_0['bool'] == 1].shape[0] + result_3[result_3['bool'] == 1].shape[0]

def get_original_number(path):
    result_0 = read_csv(os.path.join(path, '0.csv'))
    result_3 = read_csv(os.path.join(path, '3.csv'))

    shot_0_0301, shot_0_0613 = result_0['0301_origin_acc'], result_0['0613_origin_acc']
    shot_3_0301, shot_3_0613 = result_3['0301_origin_acc'], result_3['0613_origin_acc']

    return shot_0_0301.to_numpy(), shot_0_0613.to_numpy(), shot_3_0301.to_numpy(), shot_3_0613.to_numpy()

def cal(args):
    num = 0
    adv_dataset = ["sst2", "qqp", "mnli", "qnli", "rte"]
    adv_results = [None for i in adv_dataset]
    bgn_results = [None for i in adv_dataset]
    # attack_list = ['textfooler', 'textbugger', 'bertattack', 'deepwordbug', 'checklist', 'stresstest', 'semantic',]
    for path in glob.glob(os.path.join(args.path, "*")):
        # print(path)
        # get_acc(path)
        if 'adv' in path and '-mm' not in path:
            dataset = path.split('adv_')[-1].split('/')[0]
            shot_0_0301, shot_0_0613, shot_3_0301, shot_3_0613 = get_original_number(path)
            adv_results[adv_dataset.index(dataset)] = np.array([shot_0_0301, shot_0_0613, shot_3_0301, shot_3_0613])

        elif path.split('/')[2].lower() in adv_dataset:
            shot_0_0301, shot_0_0613, shot_3_0301, shot_3_0613 = get_original_number(path)
            bgn_results[adv_dataset.index(path.split('/')[2].lower())] = np.array([shot_0_0301, shot_0_0613, shot_3_0301, shot_3_0613])
            
        if 'adv' not in path:
            beee, woee = get_number(path)
            num += beee

        # print("Done")

    adv_results = np.array(adv_results)
    bgn_results = np.array(bgn_results)

    drp_results = bgn_results - adv_results

    ou = drp_results[:, ::2]
    ji = drp_results[:, 1::2]

    re = (ou > ji).sum()
    total = (ou > ji).sum() + (ou <= ji).sum()

    print(f"{total - re} / {total}")
    print(f"{num} / 168")

def get_acc(filepath):
    name = 't5'
    path = os.path.join(filepath,f"*_{name}.log")
    result_0 = np.zeros((7, 6))
    result_3 = np.zeros((7, 6))
    model_list = ['0301', '0613']
    attack_list = ['textfooler', 'textbugger', 'bertattack', 'deepwordbug', 'checklist', 'stresstest', 'semantic',]
    for file in glob.glob(path):
        with open(file, 'r') as r:
            content = r.read()
            mat_rob = re.findall(r'INFO - (\d+\.?\d*)%', content)
            mat_drp = re.findall(r'dropped acc (.*)%', content)

            if len(mat_rob) > 2:
                origin_acc = float(mat_rob[0])
                attack_acc = float(mat_rob[2])
                droped_acc = float(mat_drp[0])
            else:
                origin_acc = float(mat_rob[0])
                attack_acc = float(mat_rob[1])
                droped_acc = float(mat_drp[0])

        model, attack, shot = file.split("_")[-8].split('/')[-1], file.split("_")[-7], file.split("_")[-3]
        if shot == '0':
            result_0[attack_list.index(attack), model_list.index(model)*3+0] = origin_acc
            result_0[attack_list.index(attack), model_list.index(model)*3+1] = attack_acc
            result_0[attack_list.index(attack), model_list.index(model)*3+2] = droped_acc
        else:
            result_3[attack_list.index(attack), model_list.index(model)*3+0] = origin_acc
            result_3[attack_list.index(attack), model_list.index(model)*3+1] = attack_acc
            result_3[attack_list.index(attack), model_list.index(model)*3+2] = droped_acc

    result_0 = pd.DataFrame(result_0, index=attack_list, columns=['0301_origin_acc', '0301_attack_acc', '0301_dropped_acc', '0613_origin_acc', '0613_attack_acc', '0613_dropped_acc'])
    result_3 = pd.DataFrame(result_3, index=attack_list, columns=['0301_origin_acc', '0301_attack_acc', '0301_dropped_acc', '0613_origin_acc', '0613_attack_acc', '0613_dropped_acc'])

    result_0.to_csv(os.path.join(filepath, f"0_{name}.csv"))
    result_3.to_csv(os.path.join(filepath, f"3_{name}.csv"))

def main(args):
    for path in glob.glob(os.path.join(args.path, "c_adv_*")):
        print(path)
        get_acc(path)

def rm(filepath):
    import stat
    name = 'tl2'
    path = os.path.join(filepath,f"*_{name}.csv")
    for file in glob.glob(path):
        print(file)
        # os.chmod(file, stat.S_IWRITE)
        os.remove(file)

if __name__ == '__main__':
    args = get_args()
    main(args)