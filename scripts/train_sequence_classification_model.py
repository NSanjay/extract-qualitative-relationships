from __future__ import absolute_import

import argparse
import re
import json
import logging
import os
import random
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)20s() -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class QuartzCandidateExample(object):
    """A single training/test example for the Quartz dataset."""
    def __init__(self,
                 candidate_id,
                 knowledge_paragraph,
                 hypothesis,
                 label):
        self.candidate_id = candidate_id
        self.knowledge_paragraph = knowledge_paragraph
        self.hypothesis = hypothesis
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        str_list = [
            "para_id: {}".format(self.candidate_id),
            "knowledge_paragraph: {}".format(self.knowledge_paragraph),
            "hypothesis: {}".format(self.hypothesis)
        ]

        if self.label is not None:
            str_list.append("label: {}".format(self.label))

        return ", ".join(str_list)

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label

def read_quartz_candidates_examples(input_file, is_training):
    logger.info("inside read examples")
    with open(input_file, 'r', encoding='utf-8') as json_file:
        lines = [json.loads(line) for line in json_file]

    examples = [
        QuartzCandidateExample(
            #para_id = re.sub(r'[A-Za-z\-]','',line["id"].replace("flip","1")),
            candidate_id= line["id"],
            knowledge_paragraph = line["premise"],
            hypothesis = line["hypothesis"],
            label = int(line["label"])

        ) for line in lines
    ]
    logger.info("len::{}:::".format(len(examples)))
    return examples

tokenmap = {}

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.

    # - [CLS] Knowledge [SEP] question [SEP] answer [SEP]


    features = []
    exindex = {}
    passagelens = []
    for example_index, example in tqdm(enumerate(examples)):

        if example.knowledge_paragraph not in tokenmap.keys():
            tokens_a = tokenizer.tokenize(example.knowledge_paragraph)
            tokenmap[example.knowledge_paragraph] = tokens_a
        else:
            tokens_a = tokenmap[example.knowledge_paragraph]

        #tokens_b = None


        if example.hypothesis not in tokenmap.keys():
            tokens_b = tokenizer.tokenize(example.hypothesis)
            tokenmap[example.hypothesis] = tokens_b
        else:
            tokens_b = tokenmap[example.hypothesis]
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"

        passagelens.append(len(tokens_a) + len(tokens_b) + 3)

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

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
        label_id = example.label

        # sum_of_labels += label_id

        if example_index < 5:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (str(example.label), 0))
            logger.info("shapes:{},{}".format(len(input_ids),len(input_mask)))

        #exindex[example_index] = example.guid
        #choice_features = [(tokens,input_ids,input_mask,segment_ids)]

        exindex[example_index] = example.candidate_id
        features.append(
            InputFeatures(example_index,
                          input_ids,
                          input_mask,
                          segment_ids,
                          label_id))

    #logger.info("Maximum Knowledge length:: {}".format(maximum_knowledge_length))
    #logger.info("Maximum Question length:: {}".format(maximum_question_length))
    #logger.info("Maximum Answer length:: {}".format(maximum_answer_length))
    logger.info("Passage Token Lengths Distribution {} {} {} {} {}".format(passagelens[-1], np.percentile(passagelens, 50),
          np.percentile(passagelens, 90), np.percentile(passagelens, 95), np.percentile(passagelens, 99)))

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
    #logger.info("out:::{}\n shape::{}".format(out,out.shape))
    #logger.info("labels:::{}\n shape::{}".format(labels, labels.shape))
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def recall(out, labels):
    #logger.info("out:::{}\n shape::{}".format(out,out.shape))
    #logger.info("labels:::{}\n shape::{}".format(labels, labels.shape))
    a=np.where(np.argmax(out, axis=1)==1)
    b=np.where(labels==1)
    return len(np.intersect1d(a, b))

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
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
                        required=True,
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
                        default=8,
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

    args = parser.parse_args()


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

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    logger.info("before model load")
    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
        cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)),
        num_labels=2)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)



    if args.do_train:

        # Prepare data loader

        train_examples = read_quartz_candidates_examples(os.path.join(args.data_dir, 'train_new_candidates.jsonl'), is_training = True)
        train_features, score_index = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        current_eval_loss,current_best_accuracy,current_tr_loss,current_best_recall = 99999, 0, 99999, 0
        for i in trange(int(args.num_train_epochs), desc="Epoch"):
            train_accuracy = 0
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                #logger.info("shapes:: {} {} {} {}".format(input_ids.shape,input_mask.shape,segment_ids.shape,label_ids.shape))
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                #tmp_train_accuracy = accuracy(logits, label_ids)

                #train_accuracy += tmp_train_accuracy

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if tr_loss < current_tr_loss:
                current_tr_loss = tr_loss

            train_accuracy = train_accuracy / nb_tr_examples
            logger.info("train loss is :: {} and accuracy is {}:::".format(tr_loss,train_accuracy))

            #eval
            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = read_quartz_candidates_examples(os.path.join(args.data_dir, 'dev_new_candidates.jsonl'),
                                                                is_training=True)
                eval_features,score_index = convert_examples_to_features(
                    eval_examples, tokenizer, args.max_seq_length, True)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
                all_unique_ids = torch.tensor([f.example_id for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_unique_ids)
                # Run prediction for full data
                eval_sampler = RandomSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy, total_recall, label_1_size = 0, 0, 0, 0
                nb_eval_steps, nb_eval_examples, out_value = 0, 0, 0
                all_results, best_results = {}, {}
                for input_ids, input_mask, segment_ids, label_ids, all_unique_ids in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                        logits = model(input_ids, segment_ids, input_mask)

                    logits = logits.detach().cpu().numpy()
                    all_unique_ids = all_unique_ids.to('cpu').numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    tmp_eval_accuracy = accuracy(logits, label_ids)
                    tmp_recall = recall(logits, label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy
                    total_recall += tmp_recall

                    nb_eval_examples += input_ids.size(0)
                    input_ids = input_ids.to('cpu').numpy()
                    label_1_size += len(np.where(input_ids)[0])
                    nb_eval_steps += 1

                    for q, score, label in zip(all_unique_ids, logits, label_ids):
                        idx = score_index[q]
                        #out_label = accuracy(score,label)
                        out_label = np.argmax(score)
                        if out_label == label:
                            out_value += 1
                        all_results[idx] = (label, out_label, score[1])

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy_modif = eval_accuracy / nb_eval_examples
                total_recall = total_recall / label_1_size

                logger.info("various accuracies:: {} {}".format(out_value, eval_accuracy))
                assert out_value == eval_accuracy


                logger.info("epoch {} loss:: {} and accuracy:: {} and recall::: {}".format(i,eval_loss,eval_accuracy_modif,total_recall))
                if eval_accuracy_modif > current_best_accuracy:
                    current_best_accuracy = eval_accuracy_modif
                    current_best_recall = total_recall

                if eval_loss < current_eval_loss:
                    best_results = all_results
                    logger.info("Saving weights of epoch {} with loss::{}".format(i,eval_loss))
                    current_eval_loss = eval_loss
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                    # If we save using the predefined names, we can load using `from_pretrained`
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(args.output_dir)

                    # Load a trained model and vocabulary that you have fine-tuned
                    #model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=2)
                    #tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

                #model.to(device)

        result = {'best_eval_loss': eval_loss,
                  'nb_eval_examples': nb_eval_examples,
                  'correct_examples':eval_accuracy,
                  'best_eval_accuracy': current_best_accuracy,
                  'best_recall':current_best_recall,
                  'global_step': global_step}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        output_eval_results = os.path.join(args.output_dir, "eval_all_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        with open(output_eval_results, "w") as writer:
            all_results = dict(sorted(best_results.items(),key=lambda x: (int(x[0].split('-')[0]), -float(x[1][2]))))
            for key,val in tqdm(all_results.items(),desc='Writing Scores'):
                writer.write("{}\t{}\t{}\t{}\n".format(key,val[0],val[1],val[2]))

    if args.do_eval and not args.do_train:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
        model.to(device)
        all_results = {}
        eval_examples = read_quartz_candidates_examples(os.path.join(args.data_dir, 'test_new_candidates.jsonl'),
                                                        is_training=True)
        eval_features, score_index = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        all_unique_ids = torch.tensor([f.example_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_unique_ids)
        # Run prediction for full data
        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy,global_step = 0, 0, 0
        nb_eval_steps, nb_eval_examples, out_value = 0, 0,0
        for input_ids, input_mask, segment_ids, label_ids,all_unique_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            all_unique_ids = all_unique_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            for q, score, label in zip(all_unique_ids, logits, label_ids):
                idx = score_index[q]
                #out_label = accuracy(score, label)

                out_label = np.argmax(score)
                softmax_score = softmax(score)
                #logger.info("pred::{} label::{}".format(out_label, label))
                if out_label != label:
                    predicted_label = 1 - label
                else:
                    out_value += 1
                    predicted_label = label
                all_results[idx] = (label, predicted_label, softmax_score[1])

        logger.info("various accuracies:: {} {}".format(out_value,eval_accuracy))
        assert out_value == eval_accuracy
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy_modif = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'nb_eval_examples': nb_eval_examples,
                  'correct_examples':eval_accuracy,
                  'eval_accuracy': eval_accuracy_modif,
                  'global_step': global_step}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        output_eval_results = os.path.join(args.output_dir, "test_all_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        with open(output_eval_results, "w") as writer:
            all_results = dict(sorted(all_results.items(),key=lambda x: (int(x[0].split('-')[0]), -float(x[1][2]))))
            for key,val in tqdm(all_results.items(),desc='Writing Scores'):
                writer.write("{}\t{}\t{}\t{}\n".format(key,val[0],val[1],val[2]))

if __name__ == "__main__":
    main()