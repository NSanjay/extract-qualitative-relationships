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

filter_out_phrases = [('more the greater', 'greater'), ('less the greater', 'greater'),
                      ('more the stronger', 'stronger'), ('less the stronger', 'stronger'),
                      ('more the bigger', 'bigger'), ('less the bigger', 'bigger'),
                      ('more the smaller', 'smaller'), ('less the smaller', 'smaller'),
                      ('more the weaker', 'weaker'), ('less the weaker', 'weaker'),
                      ('more the more', 'more'),('less the more', 'more'),
                      ('less the less', 'less'), ('less more', 'more'), ('more less', 'more'),
                      ('more the less', 'less'), ('more more', 'more'),
                      ('less less', 'less'), ('more lower', 'lower'), ('less higher', 'higher'),
                      ('more higher', 'higher'), ('less lower', 'lower'), ('less greater', 'greater'),
                      ('more greater', 'greater'),
                      ('less decrease', 'decrease'), ('more increase', 'increase'), ('less decreasing', 'decrease'),
                      ('more increasing', 'increase'), ('less decreased', 'decrease'), ('more increased', 'increase'),
                      ('more many','more'),('less many','many'),('more have less','less'),('more has less','less'),('more have more','more'),('more has more','more'),
                      ('less have more','more'),('less has more','more'),('less have less','less'),('less has less','less')]

contradictory_sentences = ["more mass is caused by less mass", "less mass is caused by more mass",
                           "more speed is caused by less speed", "less speed is caused by more speed",
                           "less age is caused by more age", "more age is caused by less age",
                           "more weight is caused by less weight", "less weight is caused by more weight",
                           "more behavior is caused by less behavior", "less behavior is caused by more behavior",
                           "more smell is caused by less smell", "less smell is caused by more smell"]

split_phrases = ['more','less','greater','lesser','increasing','decreasing','additional','increase','decrease','increased','decreased','lower','higher','increases','decreases','smaller','bigger','stronger',
                 ]

filter_out_words = ['more','less','greater','lesser','increasing','decreasing','additional','increase','decrease','increased','decreased','lower','higher','increases','decreases']

priority_phrases = ['greater','lesser','additional','lower','higher','increasing','decreasing','increases','decreases','greatest','biggest','smallest','many','few']

articles = ["a","an","the",""]

articles_phrases = list([' '.join(tup).strip() for tup in itertools.product(articles,filter_out_words)])

priority_phrases = list([' '.join(tup).strip() for tup in itertools.product(articles,priority_phrases)])

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

            new_chunks = set()
            for chunk in chunks:
                if "," in chunk and any(split_word in ' '.join(chunk.split(',')[1:]) for split_word in split_phrases):
                    new_chunks.update(set(chunk.split(',')))
                    #print(chunk.split(','))
                else:
                    new_chunks.add(chunk)

            chunks = list(new_chunks)
            chunks = [chunk.strip() for chunk in chunks if chunk.strip().lower() not in articles_phrases]
            #print('new_chunks::',chunks,i)
            more_than_two_word_phrases = [chunk for chunk in chunks if len(re.split('[-\s]',chunk)) > 2]
            multi_word_phrases = [chunk for chunk in chunks if len(re.split('[-\s]',chunk)) > 1]
            if more_than_two_word_phrases:
                multi_word_phrases = [chunk for chunk in multi_word_phrases if (len(re.split('[-\s]',chunk)) == 2 and
                                                                            (not any(chunk in more_than_two_word for more_than_two_word in more_than_two_word_phrases)
                                                                             or (all(len(nlp(part_of_chunk)[0].tag_) > 1 and nlp(part_of_chunk)[0].tag_[:2] in ["NN"] for part_of_chunk in re.split('[-\s]',chunk)))))
                                  or len(re.split('[-\s]',chunk)) > 2]
            if len(multi_word_phrases) == 1:
                multi_word_phrases = [chunk for chunk in chunks if len(re.split('[-\s]', chunk)) > 1]
            multi_word_phrases += [chunk for chunk in chunks if len(re.split('[-\s]',chunk)) == 1 and not any(chunk in multi_word for multi_word in multi_word_phrases)]
            #print("initial chunks:::",multi_word_phrases)
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
            sentences = {'{} {} is caused by {} {}.'.format(val[0], permutation[0], val[1], permutation[1]) for
                         permutation in all_permutations for val in more_less_tup if permutation[0] not in permutation[1] and permutation[1] not in permutation[0]}

            #print("sentences len::{}".format(len(sentences)))
            new_sentences = set()
            for sentence in sentences:

                    # pattern = re.compile(re.escape(tup[0]), re.IGNORECASE)
                    # sentence = re.sub(pattern, tup[1], sentence)
                sentence_split = sentence.split(' is caused by ')


                if sentence_split[0].split(' ')[1:] == sentence_split[1].split(' ')[1:] and sentence_split[0].split(' ')[0] != sentence_split[1].split(' ')[0]:
                    print("contradictory sentence::",sentence)
                    continue

                # matches_1 = re.findall("|".join(priority_phrases),sentence_split[0])
                # if matches_1:
                #     sentence_split[0] = ' '.join(sentence_split[0].split(' ')[1:])
                # matches_2 = re.findall("|".join(priority_phrases), sentence_split[1])
                # if matches_2:
                #     sentence_split[1] = ' '.join(sentence_split[1].split(' ')[1:])

                sentence = sentence_split[0] + " is caused by " + sentence_split[1]
                for tup in filter_out_phrases:
                    pattern = re.compile(re.escape(tup[0]), re.IGNORECASE)
                    sentence = re.sub(pattern, tup[1], sentence)
                    new_sentence_split = sentence.split(' is caused by ')

                    # if new_sentence_split[0].split(' ')[0] not in priority_phrases+['more','less'] or new_sentence_split[1].split(' ')[0] not in priority_phrases +['more','less']:
                    #     print("bad sentence::::",sentence)

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
                new_sentences.add(sentence)
            # print(len(sentences))

            sentences = list(new_sentences)
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
                         and len(re.split('[-\s]',sentence.split(' is caused by ')[0])) > 1 and len(re.split('[-\s]',sentence.split(' is caused by ')[1])) > 1
                         and re.split('[-\s]',sentence.split(' is caused by ')[0]) not in articles_phrases and re.split('[-\s]',sentence.split(' is caused by ')[1]) not in articles_phrases
                         and ' '.join(sentence.split(' is caused by ')[0].split(' ')[1:]) not in articles_phrases
                         and ' '.join(sentence.split(' is caused by ')[1].split(' ')[1:]) not in articles_phrases]
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