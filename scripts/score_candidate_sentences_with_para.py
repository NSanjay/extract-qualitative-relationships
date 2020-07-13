from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import re
import json
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, uuid, input_ids, input_mask, segment_ids,label_id=None):
        self.uuid = uuid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        #self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class BoWProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_score_examples(self, data_dir, fname):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, fname)), "score")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1] + " . " + line[2]
            text_b = line[-1]
            label = 0
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class CustomProcessor(DataProcessor):
    stage_name_pattern = re.compile('\:\:(.*?)\:\:')
    w_patterns = re.compile('What|what|When|when|Why|why|Will|will|How|how|they are')

    def get_score_examples(self, data_dir, fname):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, fname)), "score")

    def get_score_examples_comparison_indicator(self, data_dir, fname):
        """See base class."""
        return self._create_examples_modified(
            self._read_jsonl(os.path.join(data_dir, fname)), "score")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):

            premise_text = line["premise"]
            #modified_premise_text = re.sub(self.stage_name_pattern, "", premise_text)
            #modified_premise_text = re.sub(self.w_patterns, "", modified_premise_text)
            hypothesis_text = line["hypothesis"]
            #hypothesis_text = re.sub(self.w_patterns, "", hypothesis_text)
            #a_label = int(line["label"])

            #sentences = modified_premise_text.split('.')

            id = line["id"]

            examples.append(
                    InputExample(guid=id, text_a=premise_text, text_b=hypothesis_text))
            # print("16th sentence::",sentences[16])

        return examples

    def _create_examples_modified(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            a_label = int(line["label"])
            q_type = line["type"]
            if a_label == 0 and q_type != "qLookup":
                # print("discontinue")
                continue
            sentence_number = 0
            premise_text = line["premise"]
            modified_premise_text = re.sub(self.stage_name_pattern, "", premise_text)
            modified_premise_text = re.sub(self.w_patterns, "", modified_premise_text)
            hypothesis_text = line["hypothesis"]
            hypothesis_text = re.sub(self.w_patterns, "", hypothesis_text)

            sentences = modified_premise_text.split('.')

            for j, sentence in enumerate(sentences):
                guid = "" + str(sentence_number) + "\t" + str(i) + "\t" + str(len(sentences)) + "\t" + str(a_label)
                text_a = sentence
                text_b = hypothesis_text
                label = a_label
                sentence_number += 1
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            # print("16th sentence::",sentences[16])

        return examples

    def _create_examples_split(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            a_label = int(line["label"])
            q_type = line["type"]
            if a_label == 0 and q_type != "qLookup":
                # print("discontinue")
                continue
            sentence_number = 0
            premise_text = line["premise"]
            the_id = int(line["id"])
            modified_premise_text = re.sub(self.stage_name_pattern, "", premise_text)
            modified_premise_text = re.sub(self.w_patterns, "", modified_premise_text)
            hypothesis_text = line["hypothesis"]
            hypothesis_text = re.sub(self.w_patterns, "", hypothesis_text)

            sentences = modified_premise_text.split('.')

            for j, sentence in enumerate(sentences):
                guid = "" + str(sentence_number) + "\t" + str(i) + "\t" + str(len(sentences)) + "\t" + str(a_label)
                text_a = sentence
                text_b = hypothesis_text
                label = a_label
                sentence_number += 1
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            # print("16th sentence::",sentences[16])

        return examples

    def _read_jsonl(self, input_file, quotechar=None):
        """Reads a jsonl file."""
        with open(input_file, 'r', encoding='utf-8') as json_file:
            # line = list(json_file)[2]
            lines = []
            for line in json_file:
                json_text = json.loads(line)
                # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                lines.append(json_text)
            return lines


tokenmap = {}


#def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    exindex = {}
    passagelens = []

    sum_of_labels = 0

    for (ex_index, example) in tqdm(enumerate(examples), desc="Tokenizing:"):
        if example.text_a not in tokenmap.keys():
            tokens_a = tokenizer.tokenize(example.text_a)
            tokenmap[example.text_a] = tokens_a
        else:
            tokens_a = tokenmap[example.text_a]

        tokens_b = None
        if example.text_b:
            if example.text_b not in tokenmap.keys():
                tokens_b = tokenizer.tokenize(example.text_b)
                tokenmap[example.text_b] = tokens_b
            else:
                tokens_b = tokenmap[example.text_b]
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"

            passagelens.append(len(tokens_a) + len(tokens_b) + 3)

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        #label_id = example.label

        #sum_of_labels += label_id

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (str(example.label), 0))

        exindex[ex_index] = example.guid
        features.append(
            InputFeatures(uuid=ex_index,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))

    print("Passage Token Lengths Distribution", passagelens[-1], np.percentile(passagelens, 50),
          np.percentile(passagelens, 90), np.percentile(passagelens, 95), np.percentile(passagelens, 99))
    return features, exindex


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def get_dataloader(train_features, train_batch_size):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    #all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_unique_ids = torch.tensor([f.uuid for f in train_features], dtype=torch.long)
    #train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_unique_ids)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_unique_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    return train_dataloader


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--data_dir",
    #                     default='./',
    #                     type=str,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--fname",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="Score file Name")
    # parser.add_argument("--bert_model", default="bert-large-cased", type=str,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                          "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
    #                          "bert-base-multilingual-cased, bert-base-chinese.")
    # parser.add_argument("--task_name",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The name of the task to train.")
    # parser.add_argument("--output_dir",
    #                     default='./',
    #                     type=str,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    #
    # ## Other parameters
    # parser.add_argument("--max_seq_length",
    #                     default=128,
    #                     type=int,
    #                     help="The maximum total input sequence length after WordPiece tokenization. \n"
    #                          "Sequences longer than this will be truncated, and sequences shorter \n"
    #                          "than this will be padded.")
    # parser.add_argument("--do_lower_case",
    #                     action='store_true',
    #                     help="Set this flag if you are using an uncased model.")
    # parser.add_argument("--eval_batch_size",
    #                     default=128,
    #                     type=int,
    #                     help="Total batch size for eval.")
    # parser.add_argument('--seed',
    #                     type=int,
    #                     default=42,
    #                     help="random seed for initialization")

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--task_name",
                         default=None,
                         type=str,
                         required=True,
                         help="The name of the task to train.")
    parser.add_argument("--fname",
                        default=None,
                        type=str,
                        required=True,
                        help="Score file Name")

    args = parser.parse_args()

    processors = {
        "bow": BoWProcessor,
        "custom": CustomProcessor
    }

    num_labels_task = {
        "bow": 2,
        "custom": 2
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if args.output_dir and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    #label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # score_examples = None

    score_examples = processor.get_score_examples(args.data_dir, args.fname)

    # Prepare model
    # output_model_file = os.path.join(args.output_dir, "best_model.bin")
    # model_state_dict = torch.load(output_model_file)
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
            cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),'distributed_{}'.format(args.local_rank)),
                                                          num_labels=num_labels)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    score_features, score_index = convert_examples_to_features(
        score_examples, args.max_seq_length, tokenizer)

    score_dataloader = get_dataloader(score_features, args.eval_batch_size)

    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0
    tq = tqdm(score_dataloader, desc="Scoring:::")
    all_results = {}
    for input_ids, input_mask, segment_ids, unique_ids in tq:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        #label_ids = label_ids.to(device)
        unique_ids = unique_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        logits = logits.detach().cpu().tolist()
        #label_ids = label_ids.detach().cpu().tolist()
        unique_ids = unique_ids.detach().cpu().tolist()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

        for q, scores in zip(unique_ids, logits):
            idx = score_index[q]
            scores = softmax(scores)
            score = scores[1]
            try:
                all_results[idx] = -float(score)
            except:
                logger.info("Not a number::",score)
                all_results[idx] = score

    outfd = open(args.data_dir + args.fname + "-score.tsv", "w")
    all_results = dict(sorted(all_results.items(), key=lambda x: (int(x[0].split('-')[0]), x[1])))
    for key, val in tqdm(all_results.items(), desc="Writing ScoreFile:"):
        outfd.write("%s\t%s\n" % (key, val))
    outfd.close()


if __name__ == "__main__":
    main()


