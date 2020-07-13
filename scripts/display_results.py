import argparse
from pathlib import *
import json
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)20s() -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)

def main():
    dir = '../data/QuaRTz-dataset-v1'
    file = 'eval_results.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/QuaRTz-dataset-v1", help="Input Directory")
    parser.add_argument("--output_dir", type=str, default="../test_model", help="Input Directory")
    parser.add_argument('--file_name',default='test_new_candidates.jsonl',type=str)
    parser.add_argument('--out_file_name', default='test_all_results.txt', type=str)
    args = parser.parse_args()

    with open(Path(args.data_dir,args.file_name),'r') as candidates_file:
        candidates_dict = {json.loads(line)[key] : line for line in candidates_file for key in json.loads(line) if key == "id" }
        #print(candidates_dict)

    i = 0
    j = 0
    with open(Path(args.output_dir,args.out_file_name),'r') as eval_file, \
        open(Path(args.output_dir,'results.jsonl'),'w+') as out_file:
        for k,line in enumerate(eval_file,start=1):
            # id = line.split('\t')[0]
            # json_object = json.loads(candidates_dict[id])
            # json_object["predicted_label"] = line.split('\t')[2].strip()
            # json_object["actual_label"] = line.split('\t')[1]
            # del json_object["label"]
            # out_file.write(json.dumps(json_object)+"\n")

            id = line.split('\t')[0]
            if int(id.split('-')[0]) != i:
                i += 1
                #print("i:::",i,"id:::",id)
                if int(line.split('\t')[1]) == 1:
                    j += 1
                    assert (k-1) % 8 == 0,"value of i::{} and k::{}".format(i,k)
                json_object = json.loads(candidates_dict[id])
                to_write_json_object = {}
                to_write_json_object["premise"] = json_object["premise"]
                to_write_json_object["top_actual_hypothesis"] = json.loads(candidates_dict["-".join(id.split('-')[:3])+"-pos-0"])["hypothesis"]
                to_write_json_object["top_predicted_hypothesis"] = json_object["hypothesis"]
                # json_object["predicted_label"] = line.split('\t')[2].strip()
                # json_object["actual_label"] = line.split('\t')[1]
                to_write_json_object["is_actual_predicted_at_top"] = bool(int(line.split('\t')[1]))
                out_file.write(json.dumps(to_write_json_object)+"\n")

        total_accuracy = j / i
        logger.info("number_of_correct::: {}, number_of_examples:::: {}, top accuracy {}".format(j,i,total_accuracy))

if __name__ == '__main__':
    main()
