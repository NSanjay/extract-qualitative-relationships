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
    parser.add_argument("--output_dir", type=str, default="../test_mcq_model", help="Input Directory")
    parser.add_argument('--file_name',default='test_mcq.jsonl',type=str)
    parser.add_argument('--out_file_name', default='test_all_results.txt', type=str)
    args = parser.parse_args()

    with open(Path(args.data_dir,args.file_name),'r') as candidates_file:
        candidates_dict = {json.loads(line)[key] : line for line in candidates_file for key in json.loads(line) if key == "id" }
        #print(candidates_dict)

    i = 0
    j = 0
    with open(Path(args.output_dir,args.out_file_name),'r') as eval_file, \
        open(Path(args.output_dir,'results.jsonl'),'w+') as out_file, \
        open(Path(args.data_dir,args.file_name),'r') as candidates_file:

        for line, candidate in zip(eval_file, candidates_file):
        #for line in eval_file:

            i += 1
            parts_of_line = line.split('\t')
            candidate = json.loads(candidate)
            id = parts_of_line[0]
            actual_label = int(parts_of_line[1])
            predicted_label = int(parts_of_line[2])
            json_obj = {}
            json_obj["id"] = id
            json_obj["premise"] = candidate["premise"]
            json_obj["actual_option"] = candidate["stem"][actual_label]["text"]
            json_obj["predicted_option"] = candidate["stem"][predicted_label]["text"]
            json_obj["is_correct"] = "Yes" if actual_label == predicted_label else "No"
            out_file.write(json.dumps(json_obj)+"\n")
            if actual_label == predicted_label:
                j += 1

        logger.info("number of correct examples:: {}, number of examples:: {}, accuracy:: {}".format(j,i,j/i))

if __name__ == '__main__':
    main()