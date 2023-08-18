import re
import random
import pickle
import argparse
import unicodedata
from itertools import chain
from unidecode import unidecode

from nltk.corpus import stopwords
italian_stopwords = stopwords.words("italian")
english_stopwords = stopwords.words("english")

admitted_tags = {"LOC", "PERSON", "GPE", "ORG", "FAC", "NORP"}
not_admitted_languages = {"tir"}
stops = set ( italian_stopwords + english_stopwords )
lng_trg = ""


# ====== FILE PROCESSING FUNCTIONS ====== #
def find_equal_prefix(str1, str2):
    equal_prefix = ""
    for char1, char2 in zip(str1, str2):
        if char1 == char2:
            equal_prefix += char1
        else:
            break

    last_slash_index = equal_prefix.rfind("/")
    if last_slash_index != -1:
        equal_prefix = equal_prefix[:last_slash_index+1]
    
    return equal_prefix

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

def compare_lists_length(list1, list2):
    # Se una delle due stringhe è 3 volte la lunghezza dell'altra, la combinazione verrà scartata.
    # Questo per evitare casi di palese non corrispondenza
    if len(list1) >= 3 * len(list2) or len(list2) >= 3 * len(list1):
        return True
    else:
        return False
    
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

def transform_string(input_string):

    global lng_trg
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
    transformed_string = re.sub(r'([^a-zA-Z\d])\s+\1(\s+\1)*', lambda m: m.group(0).replace(" ", "") if m.group(0).isascii()
                                                                  else m.group(0).replace(" ", " ") , transformed_string)

    #merge
    transformed_string = re.sub(r'. ([,;:])', r'.\1', transformed_string)

    #move situations like U. K. , S. A. togheter
    transformed_string = re.sub(r'([A-Z]\.) ([A-Z]\.)', r'\1\2', transformed_string)

    #merge links (.com, .js etc)
    transformed_string = re.sub(r"(\.) ([A-z]{2,3})", r"\1\2", transformed_string)

    # # uppercase letter followed by a dot and space and a number
    # modified_string = re.sub(r'([A-Z]*)\.\s+(\d+)', r'\1.\2', input_string)
    
    if lng_trg == "ur":
        transformed_string = re.sub(r'(\w)\s+([ًٌٍَُِْ])\s+(\w)', r'\1\2\3', transformed_string)
    
    return transformed_string



# ====== DNTS PROCESSING FUNCS ====== #
def levenshtein_distance(s1, s2):
    m = len(s1)
    n = len(s2)

    # Create a matrix to store the dynamic programming values
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column of the matrix
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Compute the minimum edit distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                               dp[i][j - 1] + 1,  # Insertion
                               dp[i - 1][j - 1] + 1)  # Substitution

    # Return the minimum edit distance
    return dp[m][n]

def dnt_augment(sentence1: str,
                sentence2: str,
                augment_percent: int,
                to_sample: list) -> str:
    
    res1 = sentence1
    res2 = sentence2
    
    dnt_tags = re.findall(r"\$\{DNT0\}\d+", sentence1)
    # highest_dnt = max(int(el) for el in re.findall(r"\$\{DNT0\}(\d+)", sentence1))
    
    to_sample = random.sample(range(50), k = 50)
    replacements = { re.escape(el): el + " ${DNT0}" + str(to_sample.pop()) for el in dnt_tags }
    
    for old, new in replacements.items():
        current_prob = random.random()
        if augment_percent > 0 and current_prob > augment_percent:
            res1 = re.sub(old, new, res1)
            res2 = re.sub(old, new, res2)
    
    return res1, res2 

def get_entities_align(src_sentence: tuple):
    res = list()
    
    idx = 0
    while idx < len(src_sentence):
        src_word, src_tag = src_sentence[idx]
        
        #admitted_tags.add("PRODUCT")
        if src_tag.startswith("B-") and src_tag[2:] in admitted_tags:
            entity: list = [src_sentence[idx][0]]
            entity_words_idx: list = [idx]
            
            # link the whole entity by B/I tags
            j = idx + 1
            while j < len(src_sentence) and src_sentence[j][1].startswith("I-"):
                entity.append(src_sentence[j][0])
                entity_words_idx.append(j)
                j += 1
                
            res.append( (entity, src_tag, entity_words_idx) )
        # Se la parola non ha un tag utile, ci spostiamo alla successiva 
        idx += 1
    return res

def replace_entities(idx, src, trg, src_tag, src_entity_list, entity_words_idx, to_sample, found_als):
    global lng_trg
    global not_admitted_languages
    
    if len(src_entity_list) > 1:
        
        """
        Check links related to a specific "B-" starting entity.
        If there's any and the first (smallest) one it's equal to the source_tag:
        - Create the whole entity as string type, put it inside the original source[idx]['word'] and delete all relatives 'I-' indexes from dictionary.
        - Extract a random 'POP', substitute in SRC dictionary.
        - Return the link and the .pop() element
        """

        links = src[entity_words_idx[0]]["to"]
        
        if links:
            link = links[0]
            if link in trg:
                trg_word, trg_tag, _ = trg[link].values()
                distance = levenshtein_distance(unidecode(src_entity_list[0]), unidecode(trg_word))
                
                if (trg_tag == src_tag): # or distance < 5
                    
                    # explicit SRC entity
                    new_word_src = " ".join(el for el in src_entity_list)

                    #explicit TRG entity
                    trg_entity: str = trg[link]["word"]
                    trg_entity_list: list[str] = [trg_entity]
                    trg_entity_idx: list[int] = [link]

                    i = link + 1
                    cont = 0
                    while i < len(trg) and cont < len(src_entity_list)-1 and i in trg.keys():
                        middle_distance = levenshtein_distance(trg[i-1]["word"].lower(), src_entity_list[cont].lower())
                        
                        if trg[i]["tag"].startswith("I-") or middle_distance < 6:                    
                            trg_entity = trg[i]["word"]
                            trg_entity_list.append(trg_entity)
                            trg_entity_idx.append(i)
                            i += 1
                            cont += 1
                        else:
                            break
                    
                    new_word_trg = " ".join(el for el in trg_entity_list)
                    if len(new_word_trg) == 1 and not new_word_trg.isalnum() \
                        or (compare_lists_length(src_entity_list, trg_entity_list)) \
                        or (get_string_type(new_word_src) != (get_string_type(new_word_trg))):
                        return False
                    
                    pop = to_sample.pop()
                    if "DNT" not in new_word_trg:
                        found_als.append(f"[{src_tag[2:]}][{new_word_src}] -> [{trg[link]['tag'][2:]}][{new_word_trg}]")

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

    elif len(src_entity_list) == 1 and len(src_entity_list[0]) > 1:
        
        """
        Check links related to a specific entity;
        If there are links, get the first (smallest) one and check if it's equal to the source_tag's one.
        In that case replace both SRC and TRG dictionary's word with 'POP'
        """
        src_word = src_entity_list[0]
        
        entity_distances = list()
        links = src[idx]["to"] #links [SRC ENTITY] -> [TRG ENTITIES]
        if links:
            for link in links:
                if link in trg:
                    trg_word, trg_tag= trg[link]["word"], trg[link]["tag"]
                    
                    # if we process very low resource language, just get the first link
                    if lng_trg in not_admitted_languages:
                        if "{DNT0}" not in trg_word and trg_word.isalnum():
                            entity_distances.append((0,
                                                 link, 
                                                 trg_word, 
                                                 trg_tag))
                        break
                    
                    # otherwise keep parsing them all
                    distance = levenshtein_distance(unidecode(src_word.lower()), unidecode(trg_word.lower()))
                    if (trg_tag == src_tag or distance < 3) and "{DNT0}" not in trg_word and trg_word.isalnum():
                        #print(f"{idx} = [{src_tag}]{src_word}:{unidecode(src_word)} -> {idx} = [{trg_tag}]{trg_word}:{unidecode(src_word)}")
                        entity_distances.append((distance,
                                                 link, 
                                                 trg_word, 
                                                 trg_tag))
                    

        #if there is any eligible entity, process the closest one             
        if entity_distances:
            entity_distances.sort(key=lambda x: x[0]) # sorting the list to get the closest word
            _, link, trg_word, trg_tag  = entity_distances[0]

            found_als.append(f"[{src_tag[2:]}][{src_word}] -> [{trg_tag[2:]}][{trg_word}]")
            #print(f"{idx} = [{src_tag}][{src_word}] -> {link} = [{trg_tag}][{trg_word}]")
                
            # extract a random number and replace in dictionary
            pop = "${DNT0}" + str(to_sample.pop())
            src[idx]['word'] = pop
            trg[link]['word'] = pop

def make_dnt_BIO(src_sentence, src_origin, trg_sentence, trg_origin, alignments, found_als, probability, augment_prob, verbosity = 0) -> tuple: 

    global stops
    global admitted_tags
    
    to_sample = random.sample(population = range(25) if 25 > len(src_sentence) else range(len(src_sentence)), 
                              k = len(src_sentence))
    
    src = {
        index: {
            "word": word,
            "tag": tag,
            "to": [tup[1] for tup in alignments if tup[0] == index]
        }
    for index, (word, tag) in enumerate(src_sentence)
    }


    trg = {
        index: {
            "word": word,
            "tag": tag,
            "to": [tup[1] for tup in alignments if tup[0] == index]
        }
        for index, (word, tag) in enumerate(trg_sentence)
    }

    # ====== PROCESSING WITH FAST ALIGN ====== #
    src_entities = get_entities_align(src_sentence)
    for entity, src_tag, entity_words_idx in src_entities:
        current_idx = entity_words_idx[0]
        
        current_prob = random.random()
        if current_prob >= probability: 
            replace_entities(current_idx, src, trg, src_tag, entity, entity_words_idx, to_sample, found_als)


    src_res = transform_string(" ".join([res["word"] for _, res in src.items()])).rstrip()
    trg_res = transform_string(" ".join([res["word"] for _, res in trg.items()])).rstrip()
    
    
    # ====== DNTS AUGMENTING ======= #
    if "{DNT0}" in src_res:
        if augment_prob > 0:
            src_res, trg_res = dnt_augment(src_res, trg_res, augment_prob, to_sample)
        src_res = src_origin + src_res
        trg_res = trg_origin + trg_res
        
        return src_res, trg_res 
    
    
    return src_origin.strip(), trg_origin.strip()



# ====== MAIN ====== #
def main(source_pavlov, target_pavlov, alignments, probability, augment_prob, verbosity):

    global lng_trg
    dnt_counts = 0
    
    with open(alignments, "r") as als, \
         open(source_pavlov[:-7], "r") as orig_source, \
         open(target_pavlov[:-7], "r") as orig_target, \
         open(source_pavlov + f".dnts5", "w") as source_out, \
         open(target_pavlov + f".dnts5", "w") as target_out:
        
        source = load_pickle(source_pavlov)
        target = load_pickle(target_pavlov)
        alignments_found = []
        align_count = 0

        #extract language
        lng_src, lng_trg = re.findall(r'\.\w{2,3}-\w{2,3}\.(\w{2,3})', source_pavlov + target_pavlov)
        print(f"SRC language -> {lng_src}\n")
        print(f"TRG language -> {lng_trg}\n")
        

        for i, (src_sentence, src_origin, trg_sentence, trg_origin, alignment) in enumerate(zip( source, orig_source, target, orig_target, als ), start = 0):

            # make alignments readable from (str) to (int)
            alignment = make_tuple(alignment.strip().split())

            found_als = list()
            src_sentence, trg_sentence  = make_dnt_BIO( src_sentence, 
                                                        src_origin,
                                                        trg_sentence, 
                                                        trg_origin,
                                                        alignment, 
                                                        found_als,
                                                        probability,
                                                        augment_prob,
                                                        verbosity )
            
            source_out.write(src_sentence + "\n")
            target_out.write(trg_sentence + "\n")
            
            dnt_counts += src_sentence.count("{DNT0}") + trg_sentence.count("{DNT0}")

            if i % 100_000 == 0:
                print(f"Iteration -> {i:,}")
            
            found_to_append = str((i, found_als))
            align_count += found_to_append.count("->")
            if found_als and verbosity: 
                alignments_found.append(str((i, found_als)))


        if verbosity:  
            out = find_equal_prefix(source_pavlov, target_pavlov)
            if alignments_found:

                with open( (out if "/" in out else "") + "alignments4.log" , "w") as f:
                    f.writelines([el + "\n" for el in alignments_found])


    print()
    print("  ================================  ")
    print(f"DNTs in each corpora = {dnt_counts // 2:,}")
    print("  ================================  ")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generates DNTs and replace them inside sentences")
    parser.add_argument("-s", "--source", help = "Path to input source pavlov file")
    parser.add_argument("-t", "--target",help = "Path to input target pavlov file")
    parser.add_argument("-a", "--alignents",help = "Path to alignments", default = "./alignments/sym.union.align")
    parser.add_argument("-p", "--probability", help = "In case a probability filter is wanted to be used", default= .0)
    parser.add_argument("-g", "--augment", help = "probability to duplicate dnts", default= .0)
    parser.add_argument("-v", "--verbosity", help = "Verbosity", default= 0)


    args = parser.parse_args()

    main(args.source, 
         args.target, 
         args.alignents,
         (1 - float(args.probability)) if args.probability else .0, 
         (1 - float(args.augment)) if args.augment else 0, 
         args.verbosity)
