import json
import argparse
from pathlib import *
import spacy
import csv
import itertools
import random
import logging
import re

random.seed(42)

nlp = spacy.load("en_core_web_lg")

filter_out_words = ['more','less','greater','lesser','increasing','decreasing','additional','increase','decrease','increased','decreased','lower','higher']

more_words = ['more','greater','the greater','many','additional','higher','increasing','increases','greatest','biggest']

less_words = ['less','lesser','the lesser','few','lower','decreasing','decreases','smallest']

articles = ["a","an","the","have","has"]

#nlp = spacy.load("en_core_web_md")

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)20s() -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def create_new_candidates(args):
    with open(Path(Path(args.data_dir).parent,'mappings.csv')) as mapping_file, \
        open(Path(args.data_dir,args.file_name.split('_')[0]+"_chunks"+".jsonl")) as chunks_file: # list composed from https://www.mobap.edu/wp-content/uploads/2013/01/list_of_adverbs.pdf
        reader = csv.DictReader(mapping_file)
        merged_dict = {row['phrase']: row['concept'] for row in reader}
        chunks_dict = {i: json.loads(line)["candidates"] for i,line in enumerate(chunks_file,start=1)}

    more_less_inverse_dict = {"more":"less","less":"more",'greater':'lesser','lower':'higher','higher':'lower','decrease':'increase',
                              'increase':'decrease','increasing':'decreasing','increased':'decreased','decreasing':'increasing','decreased':'increased','lesser':'greater',
                              'many':'few'}
    with open(Path(args.data_dir, args.file_name), 'r', encoding='utf-8') as in_file, \
            open(Path(args.data_dir, args.file_name.replace('top_ent', 'new_candidates')), 'w+',
                 encoding='utf-8') as new_ent_file:


        for j, line in enumerate(in_file):
            logger.info(j)
            line = json.loads(line)
            hypothesis = line["hypothesis"]
            premise = line["premise"]
            question_id = line["id"]

            tokens = nlp(premise)

            #chunks = chunks_dict[question_id]
            chunks = chunks_dict[int(question_id.split('-')[0])]
            multi_word_phrases = [chunk for chunk in chunks if len(re.split('[-\s]', chunk)) > 1]
            multi_word_phrases += [chunk for chunk in chunks if len(re.split('[-\s]', chunk)) == 1 and not any(
                chunk in multi_word for multi_word in multi_word_phrases)]
            chunks = [chunk for chunk in multi_word_phrases if
                      chunk not in nlp.Defaults.stop_words and chunk not in filter_out_words and chunk != premise]

            # iterable = itertools.chain(chunks,
            #                            [merged_dict[token.lemma_] for token in tokens if
            #                             token.lemma_ in merged_dict
            #                             and not (any(
            #                                 merged_dict[token.lemma_] in chunk for chunk in chunks))])

            new_json = dict(line)
            new_json["label"] = "1"
            for i in range(4):
                new_json["id"] += "-pos-" + str(i)
                new_ent_file.write(json.dumps(new_json)+"\n")
                new_json = dict(new_json)
                new_json["id"] = line["id"]

            try:
                new_json = dict(line)
                split_hypothesis = hypothesis.split(" is caused by ")
                split_hypothesis[1] = split_hypothesis[1].strip('.')
                new_possible_hypotheses = []
                hypothesis_part_1 = hypothesis.split(' ')[0]
                matches = re.findall("|".join(more_less_inverse_dict.keys()),hypothesis_part_1)
                for match in matches:
                    hypothesis_part_1 = hypothesis_part_1.replace(match,more_less_inverse_dict[match])
                if not matches:
                    search_pattern = [item[0] + " " + item[1] for item in itertools.product(articles,more_less_inverse_dict.keys())]
                    matches = re.findall("|".join(search_pattern), hypothesis_part_1)

                    for match in matches:
                        match_part_1 = match.split(' ')[1]
                        hypothesis_part_1 = hypothesis_part_1.replace(match_part_1, more_less_inverse_dict[match_part_1])
                hypothesis_1 = hypothesis_part_1 + " " + ' '.join(hypothesis.split(' ')[1:])
                #hypothesis_1 = more_less_inverse_dict[hypothesis.split(' ')[0]] + " " + ' '.join(hypothesis.split(' ')[1:])
                new_json["hypothesis"] = hypothesis_1
                new_json["label"] = "0"


            #write
                new_json["id"] += "-neg-" + "0"
                new_ent_file.write((json.dumps(new_json))+"\n")

                new_json = dict(line)

                hypothesis_part_2 = split_hypothesis[1].split(' ')[0]
                matches = re.findall("|".join(more_less_inverse_dict.keys()), hypothesis_part_2)
                for match in matches:
                    hypothesis_part_2 = hypothesis_part_2.replace(match, more_less_inverse_dict[match])

                if not matches:
                    search_pattern = [item[0] + " " + item[1] for item in itertools.product(articles,more_less_inverse_dict.keys())]
                    matches = re.findall("|".join(search_pattern), hypothesis_part_2)

                    for match in matches:
                        match_part_1 = match.split(' ')[1]
                        hypothesis_part_2 = hypothesis_part_2.replace(match_part_1, more_less_inverse_dict[match_part_1])
                hypothesis_2 = split_hypothesis[0] + " is caused by " + hypothesis_part_2 + ' ' + ' '.join(split_hypothesis[1].split(' ')[1:]) + "."

                #hypothesis_2 = split_hypothesis[0] + "is caused by " + more_less_inverse_dict[split_hypothesis[1].split(' ')[0]] +\              ' ' + ' '.join(split_hypothesis[1].split(' ')[1:])

                new_json["id"] += "-neg-" + "1"
                new_json["label"] = "0"
                new_json["hypothesis"] = hypothesis_2
                new_ent_file.write((json.dumps(new_json))+"\n")

                new_json = dict(line)
                i = 0
                while True:
                    score_dict = dict()


                    # random_noun_chunk = random.choice(list(tokens.noun_chunks))
                    #
                    # #logger.info("random chunk line 84::{} ------ {}".format(nlp(random_noun_chunk.text),nlp(''.join(split_hypothesis[1].split(' ')[1:]))))
                    # similarity_score = nlp(random_noun_chunk.text).similarity(nlp(' '.join(split_hypothesis[1].split(' ')[1:])))
                    #
                    # #print("chunk::",random_noun_chunk.text,"text::",' '.join(split_hypothesis[1].split(' ')[1:]),"sim::",nlp(random_noun_chunk.text).similarity(nlp(' '.join(split_hypothesis[1].split(' ')[1:]))))
                    # if random_noun_chunk.text not in split_hypothesis[0] and random_noun_chunk.text not in split_hypothesis[1] \
                    #         and similarity_score < 0.50:
                    #     hypothesis_3 = split_hypothesis[0].split(' ')[0] + " " + random_noun_chunk.text + " is caused by " + split_hypothesis[1]
                    #     break
                    #
                    # else:
                    #     score_dict[random_noun_chunk.text] = similarity_score

                    ################################ change here #################################
                    random_noun_chunk = random.choice(chunks)

                    sentence_1_part = ' '.join(split_hypothesis[1].split(' ')[1:])

                    if nlp(random_noun_chunk).vector_norm == 0 or nlp(sentence_1_part).vector_norm == 0:
                        i += 1
                        logger.info(
                            "random chunk line 84::{} ------ {} --- {}".format(nlp(random_noun_chunk), nlp(sentence_1_part), i))
                        if random_noun_chunk != sentence_1_part and random_noun_chunk not in sentence_1_part \
                                and sentence_1_part not in random_noun_chunk \
                                and random_noun_chunk not in split_hypothesis[0] \
                                and ' '.join(split_hypothesis[0].split(' ')[1:]) not in random_noun_chunk:
                            hypothesis_3 = split_hypothesis[0].split(' ')[0] + " " + random_noun_chunk + " is caused by " + \
                                           split_hypothesis[1] + "."
                            break
                        else:
                            if i == 20:
                                break_flag = False
                                logger.info("in i 20 3rd")
                                for token in tokens:
                                    if token.text not in sentence_1_part and token.text not in split_hypothesis[0]:
                                        hypothesis_3 = split_hypothesis[0].split(' ')[
                                                           0] + " " + token.text + " is caused by " + \
                                                       split_hypothesis[1] + "."
                                        break_flag = True
                                        break
                                if break_flag:
                                    break
                                logger.info("############################# error ###############################")
                                hypothesis_3 = split_hypothesis[0].split(' ')[
                                                   0] + " " + token.text + " is not caused by " + \
                                               split_hypothesis[1] + "."
                                break
                            continue
                    #logger.info("random chunk line 84::{} ------ {}".format(nlp(random_noun_chunk),nlp(sentence_1_part)))
                    #logger.info("line 180")
                    similarity_score = nlp(random_noun_chunk).similarity(
                        nlp(sentence_1_part))
                    #print(random_noun_chunk, sentence_1_part,similarity_score,' '.join(split_hypothesis[0].split(' ')[1:]),' '.join(split_hypothesis[0].split(' ')[1:]) in random_noun_chunk)

                    # print("chunk::",random_noun_chunk.text,"text::",' '.join(split_hypothesis[1].split(' ')[1:]),"sim::",nlp(random_noun_chunk.text).similarity(nlp(' '.join(split_hypothesis[1].split(' ')[1:]))))

                    if random_noun_chunk != sentence_1_part and random_noun_chunk not in split_hypothesis[0] and random_noun_chunk not in \
                            sentence_1_part and ' '.join(split_hypothesis[0].split(' ')[1:]) not in random_noun_chunk and sentence_1_part not in random_noun_chunk\
                            and 0.50 > similarity_score > 0.0:
                        hypothesis_3 = split_hypothesis[0].split(' ')[
                                           0] + " " + random_noun_chunk + " is caused by " + split_hypothesis[1] + "."
                        break

                    else:
                        score_dict[random_noun_chunk] = similarity_score

                    i += 1
                    #print(i)
                    if i == 20:
                        print("inside")
                        break_flag = False
                        sentence_1_part_without_stop_words = ' '.join([a for a in sentence_1_part.split(' ') if a not in nlp.Defaults.stop_words])
                        if not sentence_1_part_without_stop_words:
                            hypothesis_3 = split_hypothesis[0].split(' ')[
                                               0] + " " + ' '.join([a for a in sentence_1_part.split(' ') if a in nlp.Defaults.stop_words]) + " is caused by " + split_hypothesis[1] + "."
                            break
                        max_sim = 1.0
                        tok = ' '
                        for token in tokens:
                            if token.text == '.' or nlp(sentence_1_part_without_stop_words).vector_norm == 0 or token.vector_norm == 0:
                                continue
                            another_sim = token.similarity(nlp(sentence_1_part_without_stop_words))
                            if another_sim < max_sim:
                                max_sim = another_sim
                                tok = token.text
                            #logger.info("line 202 {} {}".format((token),another_sim))
                            if token.text not in sentence_1_part and token.text not in split_hypothesis[0] and token.vector_norm != 0 \
                                  and sentence_1_part not in token.text and ' '.join(split_hypothesis[0].split(' ')[1:]) not in token.text\
                                  and another_sim < 0.5:
                                hypothesis_3 = split_hypothesis[0].split(' ')[
                                                   0] + " " + token.text + " is caused by " + split_hypothesis[1] + "."
                                break_flag = True
                                break
                        if break_flag:
                            break
                        #random_chunk = sorted(score_dict.items(), key = lambda x: x[1])[0][0]
                        #print("highest random chunk::",random_chunk)
                        hypothesis_3 = split_hypothesis[0].split(' ')[0] + " " + tok + " is caused by " + split_hypothesis[1] + "."
                        break

                new_json["hypothesis"] = hypothesis_3

                new_json["id"] += "-neg-" + "2"
                new_json["label"] = "0"
                new_ent_file.write((json.dumps(new_json))+"\n")

                #
                new_json = dict(line)

                i = 0
                while True:
                    score_dict = dict()
                    #i = 0
                    # random_noun_chunk = random.choice(list(tokens.noun_chunks))
                    # #logger.info("random chunk line 119::{}------- {}".format(nlp(random_noun_chunk.text),nlp(' '.join(split_hypothesis[0].split(' ')[1:]))))
                    # similarity_score = nlp(random_noun_chunk.text).similarity(nlp(' '.join(split_hypothesis[0].split(' ')[1:])))
                    # #print("chunk::",random_noun_chunk.text,"text::",' '.join(split_hypothesis[0].split(' ')[1:]),"sim::",nlp(random_noun_chunk.text).similarity(nlp(' '.join(split_hypothesis[0].split(' ')[1:]))))
                    # if random_noun_chunk.text not in split_hypothesis[0] and random_noun_chunk.text not in split_hypothesis[1]  \
                    #         and similarity_score < 0.5:
                    #     hypothesis_4 = split_hypothesis[0]+ "is caused by " + split_hypothesis[1].split(' ')[0] + " " + random_noun_chunk.text
                    #     break
                    # else:
                    #     score_dict[random_noun_chunk.text] = similarity_score

                    ################################ change here #################################
                    random_noun_chunk = random.choice(chunks)
                    sentence_0_part = ' '.join(split_hypothesis[0].split(' ')[1:])
                    if nlp(random_noun_chunk).vector_norm == 0 or nlp(sentence_0_part).vector_norm == 0:
                        i += 1

                        if random_noun_chunk != sentence_0_part and random_noun_chunk not in sentence_0_part and sentence_0_part not in random_noun_chunk \
                                and random_noun_chunk not in split_hypothesis[1] \
                                and ' '.join(split_hypothesis[1].split(' ')[1:]) not in random_noun_chunk:
                            hypothesis_4 = split_hypothesis[0] + " is caused by " + split_hypothesis[1].split(' ')[
                                0] + " " + random_noun_chunk + "."
                            break
                        else:
                            if i == 20:
                                logger.info("in i 20 4th")
                                break_flag = False
                                for token in tokens:
                                    #logger.info("line 256")
                                    if token.text not in sentence_0_part and token.text not in split_hypothesis[1]:
                                        hypothesis_4 = split_hypothesis[0] + " is caused by " + \
                                                       split_hypothesis[1].split(' ')[
                                                           0] + " " + token.text + "."
                                        break_flag = True
                                        break
                                if break_flag:
                                    break
                                hypothesis_4 = split_hypothesis[0] + " is not caused by " + \
                                               split_hypothesis[1].split(' ')[
                                                   0] + " " + token.text + "."
                                break
                            continue


                    #logger.info("random chunk line 119::{}------- {}".format(nlp(random_noun_chunk),nlp(sentence_0_part)))
                    #logger.info("line 273")
                    similarity_score = nlp(random_noun_chunk).similarity(
                        nlp(sentence_0_part))
                    # print("chunk::",random_noun_chunk.text,"text::",' '.join(split_hypothesis[0].split(' ')[1:]),"sim::",nlp(random_noun_chunk.text).similarity(nlp(' '.join(split_hypothesis[0].split(' ')[1:]))))

                    if random_noun_chunk != sentence_0_part and random_noun_chunk not in split_hypothesis[0] and random_noun_chunk not in split_hypothesis[1] \
                            and sentence_0_part not in random_noun_chunk and sentence_1_part not in random_noun_chunk \
                            and 0.5 > similarity_score > 0.0:
                        hypothesis_4 = split_hypothesis[0] + " is caused by " + split_hypothesis[1].split(' ')[
                            0] + " " + random_noun_chunk + "."
                        break
                    else:
                        score_dict[random_noun_chunk] = similarity_score
                    i += 1
                    if i == 20:
                        #print("inside")
                        break_flag = False
                        sentence_0_part_without_stop_words = ' '.join(
                            [a for a in sentence_0_part.split(' ') if a not in nlp.Defaults.stop_words])
                        if not sentence_0_part_without_stop_words:
                            hypothesis_4 = split_hypothesis[0] + " is caused by " + split_hypothesis[1].split(' ')[
                                0] + " " + ' '.join([a for a in sentence_0_part.split(' ') if a in nlp.Defaults.stop_words]) + "."
                            break

                        max_sim = 1.0
                        tok = ' '
                        for token in tokens:
                            #logger.info("line 290")
                            if token.text == '.' or nlp(sentence_0_part_without_stop_words).vector_norm == 0 or token.vector_norm == 0:
                                continue
                            another_sim = token.similarity(nlp(sentence_0_part_without_stop_words))
                            if another_sim < max_sim:
                                max_sim = another_sim
                                tok = token.text

                            if token.text not in sentence_0_part and token.text not in split_hypothesis[1] and token.vector_norm != 0 and \
                                sentence_0_part not in token.text and ' '.join(
                                split_hypothesis[1].split(' ')[1:]) not in token.text \
                                and nlp(sentence_0_part_without_stop_words) != 0 \
                                and token.similarity(nlp(sentence_0_part_without_stop_words)) < 0.5:
                                hypothesis_4 = split_hypothesis[0] + " is caused by " + split_hypothesis[1].split(' ')[
                                    0] + " " + token.text + "."
                                break_flag = True
                                break
                        if break_flag:
                            break

                        random_chunk = sorted(score_dict.items(), key = lambda x: x[1])[0][0]
                        hypothesis_4 = split_hypothesis[0] + " is caused by " + split_hypothesis[1].split(' ')[
                            0] + " " + tok + "."
                        break

                new_json["hypothesis"] = hypothesis_4

                new_json["id"] += "-neg-" + "3"
                new_json["label"] = "0"
                new_ent_file.write((json.dumps(new_json))+"\n")
                #break
            except Exception as e:
                logger.info("error in line:: {}".format(j))
                logger.info("error is:::{}".format(e))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="../data/QuaRTz-dataset-v1",help="Input Directory")
    parser.add_argument("--file_name",type=str,default="train_top_ent.jsonl",help="Input File")
    args = parser.parse_args()
    create_new_candidates(args)

if __name__ == "__main__":
    main()
