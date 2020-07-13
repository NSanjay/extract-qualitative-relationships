import argparse
from pathlib import *
import json
import pandas as pd


def retrieve_top_ids(args):
    #with open(Path(args.data_dir, args.file_name+'-score.tsv'), 'r', encoding='utf-8') as ent_score_file:

    df = pd.read_csv(Path(args.data_dir, args.file_name+'-score.tsv'), names=["id","score"], sep="\t")
    filtered_ids = []
    top_most = 0
    for i, row in df.iterrows():
        if i == top_most:
            filtered_ids.append(row["id"])
            top_most += int(row["id"].split('-')[2])

    return filtered_ids

def retrieve_top_hypotheses(args, top_ids):
    with open(Path(args.data_dir,args.file_name),'r',encoding='utf-8') as ent_file, \
            open(Path(args.data_dir, args.file_name.replace('ent','top_ent')),'w+',encoding='utf-8') as revised_file:
        i = 0
        length_of_ids = len(top_ids)
        print(length_of_ids)

        for row in ent_file:
            row = json.loads(row)
            id = row["id"]
            if id == top_ids[i]:
                revised_file.write(json.dumps(row) + "\n")
                if i == length_of_ids - 1:
                    break
                i += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/QuaRTz-dataset-v1", help="Input Directory")
    parser.add_argument("--file_name", type=str, default="train_ent.jsonl", help="Input File")
    args = parser.parse_args()
    top_ids = retrieve_top_ids(args)
    print(top_ids)
    retrieve_top_hypotheses(args, top_ids)

if __name__ == "__main__":
    main()