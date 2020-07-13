import json
import argparse
from pathlib import *

def check_annotations_span(args):
    with open(Path(args.data_dir,args.file_name),'r',encoding='utf-8') as in_file, open(Path(args.data_dir,'annotations_'+args.file_name),'w+',encoding='utf-8') as out_file:
        j = 0
        for i,line in enumerate(in_file,start=1):
            line = json.loads(line)
            para = line["para"]
            para_annotations = line["para_anno"]
            effect_prop = para_annotations["effect_prop"]
            cause_prop = para_annotations["cause_prop"]
            if effect_prop not in para or cause_prop not in para:
                new_json = {}
                j += 1
                new_json["example_number"] = i
                new_json["id"] = line["id"]
                new_json["para"] = para
                new_json["effect_prop"] = effect_prop
                new_json["cause_prop"] = cause_prop
                out_file.write(json.dumps(new_json)+"\n")
        print("number of examples without effect or cause or both:: {}".format(j))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/QuaRTz-dataset-v1", help="Input Directory")
    parser.add_argument("--file_name", type=str, default="train.jsonl", help="Input File")
    args = parser.parse_args()
    check_annotations_span(args)


if __name__ == "__main__":
    main()