import spacy
import csv
import json
import argparse
from pathlib import *
import itertools
import re
import logging

nlp = spacy.load("en_core_web_lg", disable=["ner"])
#nlp = spacy.load("en_core_web_md")
noun_verb_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VBP', 'VBN', 'VBZ']
noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']

# filter_out_phrases = [('more the greater','the greater'),('less the greater','the greater'),('more the more','the more'),('less the more','the more'),
#                      ('less the less','the less'),('less more','more'),('more less','more'),('more the less','the less'),('more more','more'),
#                      ('less less','less'),('less increas','increas'),('more decreas','decreas'),('more lower','lower'),('less higher','higher'),
#                      ('more higher','higher'),('less lower','lower'),('less greater','greater'),('more greater','greater')]

filter_out_phrases = [('more the greater', 'greater'), ('less the greater', 'greater'), ('more the more', 'more'),
                      ('less the more', 'more'),
                      ('less the less', 'less'), ('less more', 'more'), ('more less', 'more'),
                      ('more the less', 'less'), ('more more', 'more'),
                      ('less less', 'less'), ('more lower', 'lower'), ('less higher', 'higher'),
                      ('more higher', 'higher'), ('less lower', 'lower'), ('less greater', 'greater'),
                      ('more greater', 'greater'),
                      ('less decrease', 'decrease'), ('more increase', 'increase'), ('less decreasing', 'decrease'),
                      ('more increasing', 'increase'), ('less decreased', 'decrease'), ('more increased', 'increase'),
                      ('more many','more'),('less many','many')]

contradictory_sentences = ["more mass is caused by less mass", "less mass is caused by more mass",
                           "more speed is caused by less speed", "less speed is caused by more speed",
                           "less age is caused by more age", "more age is caused by less age",
                           "more weight is caused by less weight", "less weight is caused by more weight",
                           "more behavior is caused by less behavior", "less behavior is caused by more behavior",
                           "more smell is caused by less smell", "less smell is caused by more smell"]


filter_out_words = ['more','less','greater','lesser','increasing','decreasing','additional','increase','decrease','increased','decreased','lower','higher','increases','decreases']

priority_phrases = ['greater','lesser','additional','lower','higher','increasing','decreasing','increases','decreases','greatest','biggest','smallest','many','few']

def create_pos_tags(args):
    with open(Path(Path(args.data_dir).parent,
                   'mappings.csv')) as mapping_file, \
                open(Path(args.data_dir,args.file_name.split('.')[0]+"_chunks"+".jsonl")) as chunks_file:  # list composed from https://www.mobap.edu/wp-content/uploads/2013/01/list_of_adverbs.pdf
        reader = csv.DictReader(mapping_file)
        merged_dict = {row['phrase']: row['concept'] for row in reader}

        chunks_dict = {json.loads(line)["id"]:json.loads(line)["candidates"] for line in chunks_file}
        # cached_reader = csv.DictReader(cached_file)
        # cached_dict = {row["id"]:row["chunks"] for row in cached_reader}
    more_less_tup = [('more', 'more'), ('more', 'less'), ('less', 'more'), ('less', 'less')]
    #print("chunks_dict:::",chunks_dict)

    new_dict = dict()
    with open(Path(args.data_dir, args.file_name), 'r', encoding='utf-8') as in_file, \
            open(Path(args.data_dir, args.file_name.replace('.jsonl', '_ent.jsonl')), 'w+',
                 encoding='utf-8') as ent_file:
        for i, line in enumerate(in_file, start=1):
            #print("iterations::{}".format(i))
            line = json.loads(line)
            question_id = line["id"]
            para = line["para"]
            tokens = nlp(para)

            #print("para::{} chunks:: {}".format(para, list(tokens.noun_chunks)))

            chunks = chunks_dict[question_id]
            multi_word_phrases = [chunk for chunk in chunks if len(re.split('[-\s]',chunk)) > 1]
            multi_word_phrases += [chunk for chunk in chunks if len(re.split('[-\s]',chunk)) == 1 and not any(chunk in multi_word for multi_word in multi_word_phrases)]
            #print("initial chunks:::",chunks)
            chunks = [chunk for chunk in multi_word_phrases if chunk not in nlp.Defaults.stop_words and chunk not in filter_out_words and chunk != para]
            #print("chunks::",chunks)
            #chunks = [chunk for chunk in chunks if not any(chunk in another_chunk for another_chunk in chunks if chunk != another_chunk)]
            # for token in tokens:
            #     if token.lemma_ in merged_dict and not(any(merged_dict[token.lemma_] in str(chunk) for chunk in tokens.noun_chunks)):
            #         another_list.append(merged_dict[token.lemma_])
            #         print("lemma:::",token.lemma_)
            #         import sys
            #         sys.exit(0)

            #chunks = [token for token in tokens.noun_chunks if token.text not in nlp.Defaults.stop_words]
            # print("new noun chunks::{}".format(chunks))

            iterable = itertools.chain(chunks,
                                       [merged_dict[token.lemma_] for token in tokens if
                                        token.lemma_ in merged_dict
                                        and token.pos_ != 'NOUN' and not (any(
                                            merged_dict[token.lemma_] in chunk for chunk in chunks))])
            # iterable = itertools.chain(tokens.noun_chunks,another_list)
            # new_dict[question_id] = str(iterable)
            # else:
            #     iterable = cached_dict[id]
            #     iterable = iterable[1:-1].split(',')

            all_permutations = itertools.permutations(iterable, 2)
            sentences = ['{} {} is caused by {} {}.'.format(val[0], permutation[0], val[1], permutation[1]) for
                         permutation in all_permutations for val in more_less_tup if permutation[0] not in permutation[1] and permutation[1] not in permutation[0]]

            # print("sentences len::{}".format(len(sentences)))
            new_sentences = []
            for sentence in sentences:

                    # pattern = re.compile(re.escape(tup[0]), re.IGNORECASE)
                    # sentence = re.sub(pattern, tup[1], sentence)
                sentence_split = sentence.split(' is caused by ')


                if sentence_split[0].split(' ')[1:] == sentence_split[1].split(' ')[1:] and sentence_split[0].split(' ')[0] != sentence_split[1].split(' ')[0]:
                    print("contradictory sentence::",sentence)
                    continue

                matches_1 = re.findall("|".join(priority_phrases),sentence_split[0])
                if matches_1:
                    sentence_split[0] = ' '.join(sentence_split[0].split(' ')[1:])
                matches_2 = re.findall("|".join(priority_phrases), sentence_split[1])
                if matches_2:
                    sentence_split[1] = ' '.join(sentence_split[1].split(' ')[1:])

                sentence = sentence_split[0] + " is caused by " + sentence_split[1]
                for tup in filter_out_phrases:
                    pattern = re.compile(re.escape(tup[0]), re.IGNORECASE)
                    sentence = re.sub(pattern, tup[1], sentence)
                    new_sentence_split = sentence.split(' is caused by ')

                # sentence_split = sentence.split(' is caused by ')
                # if sentence_split[0].split(' ')[1].lower() in filter_out_words:
                #     sentence = ' '.join(sentence_split[0].split(' ')[1:]) + ' is caused by ' + sentence_split[1]
                #
                # sentence_split = sentence.split(' is caused by ')
                # if sentence_split[1].split(' ')[1].lower() in filter_out_words:
                #     sentence = sentence_split[0] + ' is caused by ' + ' '.join(sentence_split[1].split(' ')[1:])
                if len(re.split('[-\s]',new_sentence_split[0])) == 1 or len(re.split('[-\s]',new_sentence_split[1])) == 1:
                    #print("ill-formed sentence::", sentence)
                    continue
                new_sentences.append(sentence)
            # print(len(sentences))

            sentences = new_sentences
            if len(sentences) == 0:

                nouns = []
                for token in tokens:
                    # print(token,token.tag_)
                    if token.tag_ in noun_verb_tags and not token.is_stop:
                        nouns.append(token)
                all_permutations = itertools.permutations(nouns, 2)
                sentences = ['{} {} is caused by {} {}.'.format(val[0], permutation[0], val[1], permutation[1]) for
                             permutation in all_permutations for val in more_less_tup]
                print("new len::", len(sentences), para,chunks,i)

            sentences = [sentence for sentence in sentences if sentence not in contradictory_sentences
                         and len(re.split('[-\s]',sentence.split(' is caused by ')[0])) > 1 and len(re.split('[-\s]',sentence.split(' is caused by ')[1])) > 1]
            list_of_dicts = [dict(zip(("hypothesis", "premise", "id"),
                                      (sentence, para, str(i) + "-" + str(j) + "-" + str(len(sentences)))))
                             for j, sentence in enumerate(sentences, start=1)]

            ent_file.writelines(json.dumps(a_json) + "\n" for a_json in list_of_dicts)

    with open(Path(Path(args.data_dir).parent, "cached_chunks.csv"), mode='a', encoding='utf-8') as cached_file:
        writer = csv.DictWriter(cached_file, fieldnames=["id", "chunks"])
        writer.writerows({"id": id, "chunks": value} for id, value in new_dict.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/QuaRTz-dataset-v1", help="Input Directory")
    parser.add_argument("--file_name", type=str, default="train.jsonl", help="Input File")
    args = parser.parse_args()
    create_pos_tags(args)


if __name__ == "__main__":
    main()