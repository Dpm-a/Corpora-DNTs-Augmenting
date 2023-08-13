import os
import pickle
import argparse
import subprocess
from tqdm import tqdm

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

def load_pickle(filename):
    # create a pickle's iterator
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def main(input_source_file,
         input_target_file,
         fa_iterations):
    
    out = find_equal_prefix(input_source_file, input_target_file)

    source = load_pickle(input_source_file)
    target = load_pickle(input_target_file)
    output_file = out + "merged"
    alignments_folder = "./alignments"
    
    if not os.path.exists(alignments_folder):
        os.makedirs(alignments_folder)
    
    with open(output_file, "w") as output, \
         open(f"{alignments_folder}/forward.align", 'w') as ff, \
         open(f"{alignments_folder}/reverse.align", 'w')  as fr, \
         open(f"{alignments_folder}/symm.union.align", "w") as un:
        

        # =================== MERGING FILES =================== #
        # hello , this is an example ||| ciao , questo è un esempio
        
        for src, trg in tqdm(zip(source, target), desc = "Merging corpora for alignments"):

            output.write(" ".join([word for word, _ in src]) + " ||| " +
                         " ".join([word for word, _ in trg]) + "\n")
            
        output.close()
        print("... DONE Merging files ...\n\n... Creating Alignments ...\n")


        # =================== GENERATING ALIGNMENTS =================== #
        
        print("... Generating FORWARD Alignments ...\n")
        alignments = subprocess.run(["./fast_align", "-i", f"{output_file}", "-d", "-o", "-v", "-I", f"{fa_iterations}"], capture_output=True, text=True)
        ff.write(alignments.stdout)
        #print(">>>", alignments.stdout)
        ff.close()
        
        print("... Generating REVERSE Alignments ...\n")
        alignments = subprocess.run(["./fast_align", "-i", f"{output_file}", "-d", "-o", "-v", "-r", "-I", f"{fa_iterations}"], capture_output=True, text=True)
        fr.write(alignments.stdout)
        fr.close()

        print("... Generating UNION Alignments ...\n")
        alignments = subprocess.run(["./atools", "-i", "fa_forward", "-j", "fa_reverse", "-c", "union"], capture_output=True, text=True)
        un.write(alignments.stdout)
                    
    
    
    print(" DONE ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Make pavlov corpora suitable for Fast Align")
    parser.add_argument("-s", help = "Path to input source language file")
    parser.add_argument("-t", help = "Path to input target language file")
    parser.add_argument("-i", help = "Fast Align Training Iteration", default=5)


    args = parser.parse_args()

    main(args.s, args.t, int(args.i))