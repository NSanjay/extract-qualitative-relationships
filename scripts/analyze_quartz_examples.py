import json
import os
import csv
from pathlib import *
import argparse

def analyze_quartz_examples(args):
	with open(Path(args.data_dir,args.file_name),'r', encoding='utf-8') as json_file, \
		open(Path(args.data_dir, args.file_name.replace('.jsonl','.csv')),'w+',encoding='utf-8') as out_file:
		csv_writer = csv.writer(out_file, quoting=csv.QUOTE_NONE, escapechar='\\')
		csv_writer.writerow(["Para", "Cause", "Effect"])
		for i, line in enumerate(json_file):
			json_object = json.loads(line)
			question = json_object["question"]["stem"]
			para = json_object["para"]
			para_cause_annotation = json_object["para_anno"]["cause_prop"]
			para_effect_annotation = json_object["para_anno"]["effect_prop"]
			to_write = []
			if para_cause_annotation not in para:
				to_write.append(para_cause_annotation)
			if para_effect_annotation not in para:
				if to_write:
					to_write.append(para_effect_annotation)
				else:
					to_write = [""] + [para_effect_annotation]

			if to_write:
				csv_writer.writerow([para] + to_write)
				

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default='../data/QuaRTz-dataset-v1',type=str)
    parser.add_argument('--file_name',default="train.jsonl",type=str)

    args = parser.parse_args()

    analyze_quartz_examples(args)

if __name__ == '__main__':
    main()