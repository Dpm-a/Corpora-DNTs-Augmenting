import argparse
import re
import time
from tqdm import tqdm
import random
import pickle
from deeppavlov import configs, build_model

import deeppavlov
import transformers
from torchcrf import CRF


# !pip install deeppavlov
# !pip install transformers
# !pip install pytorch-crf


def tag_sentences(ner_output: list) -> tuple:

    # estraggo lista di tuple (word_tag) per frasi src e trg
    src_sentence = list(zip( ner_output[0][0], ner_output[1][0] ))
    trg_sentence = list(zip( ner_output[0][1], ner_output[1][1] ))
    return src_sentence, trg_sentence




def main(input_source_file,
         input_target_file,
         DISPLAY_ITERATION):

    ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

    with open(input_source_file, 'r') as source, \
         open(input_target_file, 'r') as target, \
         open(input_source_file + ".pavlov", 'wb') as os, \
         open(input_target_file + ".pavlov", 'wb') as ot:

        parsed = 0
        start_time = time.time()
        for src_sentence, trg_sentence in zip(source, target):

            output = ner_model([src_sentence.rstrip("\n"), trg_sentence.rstrip("\n")])

            # taggo frase source (it) per cercare le target tags
            src_sentence, trg_sentence = tag_sentences(output)

            pickle.dump(src_sentence, os)
            pickle.dump(trg_sentence, ot)

            parsed += 1

            if parsed % DISPLAY_ITERATION == 0:
                print(f"- {parsed = } -")
                end_time = time.time()
                elapsed_minutes = (end_time - start_time) / 60   # Convert seconds to minutes
                print(f"Time elapsed: {elapsed_minutes:.2f} minutes")

        print(f" - DONE - ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Substitute DNTs in given texts")
    parser.add_argument("-s", help="Path to input source language file")
    parser.add_argument("-t", help="Path to input target language file")
    parser.add_argument("-i", "--iterations", help="Display counter at each i'th iteration",
                        default=200)


    args = parser.parse_args()

    main(args.s, args.t, int(args.iterations))
