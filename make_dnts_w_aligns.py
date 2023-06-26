import re
import random
import pickle
import argparse
from unidecode import unidecode

from nltk.corpus import stopwords
italian_stopwords = stopwords.words("italian")
english_stopwords = stopwords.words("english")

admitted_tags = {"LOC", "PERSON", "GPE", "ORG", "FAC"}
stops = set ( italian_stopwords + english_stopwords )
discarded = []



def make_tuple(indexes_row: list):
    res = list() 
    for el in indexes_row:
        i = el.index("-")
        res.append( ( int(el[:i]), int(el[i+1:]) ) )
    return res

def load_pickle(filename):
    # create a pickle's iterator
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def are_sentences_same(sentence1, sentence2):
    words1 = set(sentence1.split())
    words2 = set(sentence2)
    return words1 == words2

def find_subsequence_indexes(sentence, subsequence):
    indexes = []
    sub_len = len(subsequence)

    # Iterate through the sentence to find matching subsequences
    for i in range(len(sentence) - sub_len + 1):
        sub_sentence = ' '.join(sentence[i:i + sub_len])
        if are_sentences_same(sub_sentence, subsequence):
            indexes.append(list(range(i, i + sub_len)))

    if indexes:
        return indexes  # Return the indexes of the found subsequences
    else:
        return False  # If no subsequences are found, return False
    
def compare_lists_length(list1, list2):
    # Se una delle due stringhe è 3 volte la lunghezza dell'altra, la combinazione verrà scartata.
    # Questo per evitare casi di palese non corrispondenza
    if len(list1) >= 3 * len(list2) or len(list2) >= 3 * len(list1):
        return True
    else:
        return False
    
def transform_string(input_string):

    # Move .,;!? and ) next to the previous word
    transformed_string = re.sub(r'\s*([.,:%;!?\])])\s*', r'\1 ', input_string)
    
    # Move - to create a unique word with the previous and next word
    transformed_string = re.sub(r'\s*-\s*', '-', transformed_string)
    
    # Move opening parenthesis to the next word
    transformed_string = re.sub(r'\(\s*', '(', transformed_string)
    transformed_string = re.sub(r'\[\s*', '[', transformed_string)
    
    # Fix the format of splitted numbers
    transformed_string = re.sub(r'(\d+([.,]\s*\d+)+)', lambda m: re.sub(r'\s*', '', m.group()), transformed_string)
    
    # Join the extracted matches with a space
    transformed_string = re.sub(r'" ([^"]+) "', r'"\1"', transformed_string)

    #merge consecutive symbols
    transformed_string = re.sub(r'([^a-zA-Z])\s+\1(\s+\1)*', lambda m: m.group(0).replace(" ", ""), transformed_string)

    # # uppercase letter followed by a dot and space and a number
    # modified_string = re.sub(r'([A-Z]*)\.\s+(\d+)', r'\1.\2', input_string)
    
    return transformed_string

def get_string_type(string):
    # Remove whitespace from the string
    string = string.strip()

    # Check if the string contains only digits
    if re.match(r'^\d+$', string):
        return "Numeric"

    # Check if the string contains only letters
    if re.match(r'^[a-zA-Z]+$', string):
        return "Alphabetic"

    # Check if the string contains only symbols/punctuation marks
    if re.match(r'^[^\w\s]+$', string):
        return "Symbols/Punctuation"

    # If none of the above conditions are met, it can be a combination of types
    return "Combination"


def get_entities(sentence):
    entities = []       # Lista per le entità trovate nel testo
    entities_idx = []   # Lista per gli indici delle parole che compongono le entità

    idx = 0
    while idx < len(sentence):
        word, tag = sentence[idx][0], sentence[idx][1]
        if tag.startswith("B-") and tag[2:] in admitted_tags:
            # Se la parola ha un tag che inizia con "B-" ed è tra quelli ammessi, allora è l'inizio di un'entità
            entity = [word]             # Lista per memorizzare le parole che compongono l'entità
            entity_words_idx = [idx]    # Lista per memorizzare gli indici delle parole che compongono l'entità

            # Collega tutte le parole dell'entità con tag "I-" successive
            j = idx + 1
            while j < len(sentence) and sentence[j][1].startswith("I-"):
                entity.append(sentence[j][0])
                entity_words_idx.append(j)
                j += 1

            entities.append(entity)
            entities_idx.append(entity_words_idx)
            idx = j     # Salta all'indice successivo dopo l'entità
        else:
            idx += 1    # Passa all'indice successivo se la parola non fa parte di un'entità

    entities_dict = {}    # Dizionario per memorizzare le entità trovate
    for entity, entity_idxs in zip(entities, entities_idx):
        entity = " ".join(entity).strip("'")   # Unisce le parole dell'entità in una stringa

        if len(entity) > 1 or (len(entity) == 1 and entity.isalnum()):
            # Verifica se l'entità ha una lunghezza maggiore di 1 o se è un singolo carattere alfanumerico
            if entity.lower() not in stops:
                # Verifica se l'entità non è una stop word
                entities_dict.setdefault(entity, []).append(entity_idxs)

    return entities_dict


def replace_entities_no_align(src_sentence, trg_sentence, to_sample, probability):
    res_src = src_sentence.copy()  # Copia la frase sorgente
    res_trg = trg_sentence.copy()  # Copia la frase destinazione

    src_entities = get_entities(src_sentence)  # Estrae le entità dalla frase sorgente
    trg_entities = get_entities(trg_sentence)  # Estrae le entità dalla frase destinazione


    to_delete_src, to_delete_trg = [], []  # Liste per memorizzare gli indici delle parole da eliminare
    if src_entities:

        # Sostituzione delle entità nella frase sorgente
        for entity, idxs in src_entities.items():
            
            current_prob = random.random()
            if current_prob >= probability:
                entity = unidecode(entity.lower()).split()  # Normalizza l'entità e la divide in singole parole
                trg_indexes = find_subsequence_indexes([unidecode(el[0].lower()) for el in res_trg], entity)
                
                # Verifica se sono state trovate sottosequenze corrispondenti nella frase destinazione
                if not trg_indexes or not idxs:
                    continue

                pop = "${DNT0}" + str(to_sample.pop())  # Genera un identificatore unico per l'entità

                # Sostituzione delle entità trovate nella frase sorgente e destinazione
                if len(idxs) > 1:
                    for lst in idxs:
                        if not lst:
                            continue
                        to_delete_s = lst[0]
                        res_src[to_delete_s] = (pop, "ENTITY")
                        if len(lst) > 1:
                            to_delete_src.extend(lst[1:])  # Memorizza gli indici delle parole da eliminare
                else:
                    to_delete_s = idxs[0][0]
                    res_src[to_delete_s] = (pop, "ENTITY")
                    if len(idxs[0]) > 1:
                        to_delete_src.extend(idxs[0][1:])  # Memorizza gli indici delle parole da eliminare

                if (len(trg_indexes)) > 1:
                    for lst in trg_indexes:
                        if not lst:
                            continue
                        to_delete_t = lst[0]
                        res_trg[to_delete_t] = (pop, "ENTITY")
                        if len(lst) > 1:
                            to_delete_trg.extend(lst[1:])  # Memorizza gli indici delle parole da eliminare

                else:
                    to_delete_t = trg_indexes[0][0]
                    res_trg[to_delete_t] = (pop, "ENTITY")
                    if len(trg_indexes[0]) > 1:
                        to_delete_trg.extend(trg_indexes[0][1:])  # Memorizza gli indici delle parole da eliminare

        if to_delete_src and to_delete_trg:
            res_src = [el if i not in to_delete_src else (el[0], "TO_SKIP") for i, el in enumerate(res_src)]  # Rimuove le parole marcate come TO_SKIP
            res_trg = [el if i not in to_delete_trg else (el[0], "TO_SKIP") for i, el in enumerate(res_trg)]  # Rimuove le parole marcate come TO_SKIP


    to_delete_src, to_delete_trg = [], []
    if trg_entities:

        # Sostituzione delle entità nella frase destinazione
        for entity, idxs in trg_entities.items():
            
            current_prob = random.random()
            if current_prob >= probability:
                entity = unidecode(entity.lower()).split()  # Normalizza l'entità e la divide in singole parole
                src_indexes = find_subsequence_indexes([unidecode(el[0].lower()) for el in res_src], entity)
                
                # Verifica se sono state trovate sottosequenze corrispondenti nella frase destinazione
                if not src_indexes or not trg_entities or (len(src_indexes) != len(idxs)):
                    continue

                pop = "${DNT0}" + str(to_sample.pop())  # Genera un identificatore unico per l'entità

                # Sostituzione delle entità trovate nella frase sorgente e destinazione
                if len(idxs) > 1:
                    for lst in idxs:
                        if not lst:
                            continue
                        to_delete_t = lst[0]
                        res_trg[to_delete_t] = (pop, "ENTITY")
                        if len(lst) > 1:
                            to_delete_trg.extend(lst[1:])  # Memorizza gli indici delle parole da eliminare
                else:
                    to_delete_t = idxs[0][0]
                    res_trg[to_delete_t] = (pop, "ENTITY")
                    if len(idxs[0]) > 1:
                        to_delete_trg.extend(idxs[0][1:])  # Memorizza gli indici delle parole da eliminare

                if (len(src_indexes)) > 1:
                    for lst in src_indexes:
                        if not lst:
                            continue
                        to_delete_s = lst[0]
                        res_src[to_delete_s] = (pop, "ENTITY")
                        if len(lst) > 1:
                            to_delete_src.extend(lst[1:])  # Memorizza gli indici delle parole da eliminare

                else:
                    to_delete_s = src_indexes[0][0]
                    res_src[to_delete_s] = (pop, "ENTITY")
                    if len(src_indexes[0]) > 1:
                        to_delete_src.extend(src_indexes[0][1:])  # Memorizza gli indici delle parole da eliminare

        if to_delete_src and to_delete_trg:
            res_trg = [el if i not in to_delete_trg else (el[0], "TO_SKIP") for i, el in enumerate(res_trg)]  # Rimuove le parole marcate come TO_SKIP
            res_src = [el if i not in to_delete_src else (el[0], "TO_SKIP") for i, el in enumerate(res_src)]  # Rimuove le parole marcate come TO_SKIP
    
    return res_src, res_trg


def replace_single(idx, trg_to_discard, src, trg, src_word, src_tag, to_sample, found_als) -> None:

    """
    Check links related to a specific entity;
    If there are links, get the first (smallest) one and check if it's equal to the source_tag's one.
    In that case replace both SRC and TRG dictionary's word with 'POP'
    """

    links = src[idx]["to"]
    if links and links[0] in trg:
        trg_word, trg_tag= trg[links[0]]["word"], trg[links[0]]["tag"]

        clear_src_word = unidecode(src_word).lower()
        clear_trg_word = unidecode(trg_word).lower()
        if (trg_tag == src_tag and "DNT" not in trg_word and trg_word.isalnum()
            and clear_src_word != clear_trg_word
            ):

            trg_words = [unidecode(word).lower() for word in trg_to_discard.split()]
            if clear_src_word in trg_words:
                return

            found_als.append(f"{idx} = [{src_tag}][{src_word}] -> {links[0]} = [{trg_tag}][{trg_word}]")
                
            # extract a random number and replace in dictionary
            pop = "${DNT0}" + str(to_sample.pop())
            src[idx]['word'] = pop
            trg[links[0]]['word'] = pop


def replace_multiple(idx, src, trg, src_tag, src_entity_list, entity_words_idx, to_sample, found_als) -> tuple:

    """
    Check links related to a specific "B-" starting entity.
    If there's any and the first (smallest) one it's equal to the source_tag:
    - Create the whole entity as string type, put it inside the original source[idx]['word'] and delete all relatives 'I-' indexes from dictionary.
    - Extract a random 'POP', substitute in SRC dictionary.
    - Return the link and the .pop() element
    """

    link = src[entity_words_idx[0]]["to"]
    
    if link:
        # print("LINK: ", link)
        link = link[0]
        if link in trg:
            # print(f"{entity = } | {trg[link]['word'] = } | {trg[link]['tag'] = } | {link = }")
            if trg[link]["tag"] == src_tag or (trg[link]["tag"] in ["PRODUCT", "ORG"] or src_tag in ["PRODUCT", "ORG"]):
                
                # explicit SRC entity
                new_word_src = " ".join(el for el in src_entity_list)

                #explicit TRG entity
                trg_entity: str = trg[link]["word"]
                trg_entity_list: list[str] = [trg_entity]
                trg_entity_idx: list[int] = [link]

                i = link + 1
                while i < len(trg):
                    if trg[i]["tag"].startswith("I-") :
                        trg_entity = trg[i]["word"]
                        trg_entity_list.append(trg_entity)
                        trg_entity_idx.append(i)
                        i += 1
                    else:
                        break
                
                new_word_trg = " ".join(el for el in trg_entity_list)
                if len(new_word_trg) == 1 and not new_word_trg.isalnum() \
                    or(compare_lists_length(src_entity_list, trg_entity_list)) \
                    or (get_string_type(new_word_src) != (get_string_type(new_word_trg))):
                    return False
                
                pop = to_sample.pop()
                if "DNT" not in new_word_trg:
                    # Replace SRC

                    src[idx] = {"word": new_word_src}
                    for i in entity_words_idx[1:]:
                        del src[i]
                    src[idx]['word'] = "${DNT0}" + str(pop)
                    src[idx]['tag'] = "ENTITY"

                    # Replace TRG
                    for i in trg_entity_idx[1:]:
                        del trg[i]
                    trg[link]['word'] = "${DNT0}" + str(pop)
                    trg[link]['tag'] = "ENTITY"

                    found_als.append(f"{entity_words_idx[0]} = [{src_tag}][{new_word_src}] -> {link} = [{trg[link]['tag']}][{new_word_trg}]")
                    return True
                
    return False


def make_dnt_BIO(src_sentence, trg_sentence, alignments, found_als, sentence_index, modality, probability):

    global stops
    global admitted_tags
    global discarded

    src_to_discard = " ".join(word for word,_ in src_sentence)
    trg_to_discard = " ".join(word for word,_ in trg_sentence)

    to_sample = random.sample(range(25) if 25 > len(src_sentence) else range(len(src_sentence)), 
                              k = len(src_sentence))
    src_sentence, trg_sentence = replace_entities_no_align(src_sentence, trg_sentence, to_sample, probability)

    if modality == "no_align":
         # Se il rateo di DNTs è troppo elevato, restituiamo la frase originale
        dnts = sum( 1 if "DNT" in word else 0 for word, _ in src_sentence)
        others = len([word for word,_ in src_sentence if len(word) > 1 and word.isalnum()])
        sentence_dnt_rateo = (dnts / others) if others > 0 else 1
        if sentence_dnt_rateo > .5:
            discarded.append(sentence_index)
            return src_to_discard, trg_to_discard, 0

        src_res = transform_string( " ".join([word 
                                                for word, tag in src_sentence
                                                if tag != "TO_SKIP"]) ).rstrip()
        trg_res = transform_string( " ".join([word 
                                                for word, tag in trg_sentence
                                                if tag != "TO_SKIP"]) ).rstrip()
        return src_res, trg_res, dnts

    
    src = { i : {"word" : word, 
                 "tag": tag, 
                 "to": [tup[1] for tup in alignments if tup[0]==i] } 
            for i, (word, tag) in enumerate(src_sentence) }
    
    trg = { i : {"word" : word, 
                 "tag": tag, 
                 "to": [tup[0] for tup in alignments if tup[1]==i] } 
            for i, (word, tag) in enumerate(trg_sentence) }
    
    idx = 0
    while idx < len(src_sentence):
        src_word, src_tag = src_sentence[idx][0], src_sentence[idx][1]
        
        if src_tag.startswith("B-") and src_tag[2:] in admitted_tags:
            entity: list = [src_sentence[idx][0]]
            entity_words_idx: list = [idx]

            # link the whole entity by B/I tags
            j = idx + 1
            while j < len(src_sentence) and src_sentence[j][1].startswith("I-"):
                entity.append(src_sentence[j][0])
                entity_words_idx.append(j)
                j += 1

            current_prob = random.random()
            if current_prob >= probability:
                if len(entity) > 1:
                    replaced = replace_multiple(idx, src, trg, src_tag, entity, entity_words_idx, to_sample, found_als)
                    if replaced: 
                        idx = j

                # if entity is only 1 word long, we find the link in trg
                elif len(entity) == 1 and entity[0].isalnum():
                    replace_single(idx, trg_to_discard, src, trg, src_word, src_tag, to_sample, found_als)
                    idx = j - 1

        # Se la parola non ha un tag utile, ci spostiamo alla successiva 
        idx += 1


    # Se il rateo di DNTs è troppo elevato, restituiamo la frase originale
    dnts = sum( 1 if "DNT" in word else 0 for word in [el["word"]for el in src.values()])
    others = len([word["word"] for word in src.values() if len(word["word"]) > 1 and word["word"].isalnum()])
    sentence_dnt_rateo = (dnts / others) if others > 0 else 1
    if sentence_dnt_rateo > .5:
        discarded.append(sentence_index)
        return src_to_discard, trg_to_discard, 0
    
    src_res = transform_string( " ".join([res["word"] 
                                          for _, res in src.items() 
                                          if res["tag"] != "TO_SKIP"]) ).rstrip()
    trg_res = transform_string( " ".join([res["word"] 
                                          for _, res in trg.items() 
                                          if res["tag"] != "TO_SKIP"]) ).rstrip()
    return src_res, trg_res, dnts



def main(source_pavlov, target_pavlov, alignments, modality, probability):

    dnt_counts = 0
    
    with open(alignments, "r") as als, \
         open(source_pavlov + f".dnts.{modality}", "w") as source_out, \
         open(target_pavlov + f".dnts.{modality}", "w") as target_out, \
         open("discarted_dnts", "w") as d:
        
        source = load_pickle(source_pavlov)
        target = load_pickle(target_pavlov)
        alignments_found = []

        for i, (src_sentence, trg_sentence, alignment) in enumerate(zip( source, target, als ), start = 0):

            # make alignments readable from (str) to (int)
            alignment = alignment.strip().split()
            alignment = make_tuple(alignment)

            found_als = list()
            src_sentence, trg_sentence , dnt_count = make_dnt_BIO(src_sentence, trg_sentence, 
                                                                    alignment, 
                                                                    found_als, i, 
                                                                    modality, probability)
            
            source_out.write(src_sentence + "\n")
            target_out.write(trg_sentence + "\n")
            if found_als: 
                alignments_found.append(str((i, found_als)))

            dnt_counts += dnt_count

            if i % 50_000 == 0:
                print(f"Iteration -> {i:,}")

        if alignments_found and modality == "align": 
            with open("alignments.log", "w") as f:
                f.writelines([el + "\n" for el in alignments_found])

        d.writelines([str(el) + "\n" for el in discarded])

    print()
    print("  ================================  ")
    print(f"DNTs in each corpora = {dnt_counts:,}")
    print()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make pavlov corpora suitable for Fast Align")
    parser.add_argument("-s", help = "Path to input source pavlov file")
    parser.add_argument("-t", help = "Path to input target pavlov file")
    parser.add_argument("-a", help = "Path to alignments", default = "./alignments/sym.union.align")
    parser.add_argument("-m", help = "Tagging Modality", choices=["align", "no_align"], default = "align")
    parser.add_argument("-p", help = "In case a probability filter is wanted to be used", default= .0)

    args = parser.parse_args()

    main(args.s, args.t, args.a, args.m, (1 - float(args.p)) if args.p else .0)