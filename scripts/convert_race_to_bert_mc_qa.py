import json
import os
from pathlib import *
import argparse
import glob

def read_race_examples(args):
    examples = []
    paths = [Path(args.data_dir,args.data_subdir,'high'), Path(args.data_dir,args.data_subdir,'middle')]
    #paths = [args.data_dir+args.data_subdir+"/high", args.data_dir+args.data_subdir+"/middle"]
    with open(Path(args.data_dir,args.data_subdir,args.data_subdir+'.jsonl'),mode="w",encoding='utf-8') as json_file:
        for path in paths:
            filenames = path.glob('**/*.txt')
            for filename in filenames:
                with open(filename, 'r', encoding='utf-8') as fpr:
                    data_raw = json.load(fpr)
                    article = data_raw['article']
                    ## for each qn
                    for i in range(len(data_raw['answers'])):
                        json_obj = {}
                        json_obj["answerKey"] = data_raw['answers'][i]
                        json_obj["id"] = data_raw["id"].split('.')[0] +'-'+str(i)
                        json_obj["para"] = article
                        json_obj["question"] = {}
                        json_obj["question"]["stem"] = data_raw['questions'][i]
                        json_obj["question"]["choices"] = [{"label":chr(65+j),"text":value} for j,value in enumerate(data_raw["options"][i])]
                        #json_obj["question"]["choices"] = 

                        #truth = ord(data_raw['answers'][i]) - ord('A')
                        #question = data_raw['questions'][i]
                        #options = data_raw['options'][i]
                        # examples.append(
                        #     RaceExample(
                        #         race_id = filename+'-'+str(i),
                        #         context_sentence = article,
                        #         start_ending = question,

                        #         ending_0 = options[0],
                        #         ending_1 = options[1],
                        #         ending_2 = options[2],
                        #         ending_3 = options[3],
                        #         label = truth))

                        json_file.write(json.dumps(json_obj)+'\n')
                        #break
                #break
            #break
                
    #return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default='../data/RACE',type=str)
    parser.add_argument('--data_subdir',default="train",type=str)

    args = parser.parse_args()

    read_race_examples(args)
if __name__ == '__main__':
    main()