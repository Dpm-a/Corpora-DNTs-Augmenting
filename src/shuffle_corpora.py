import random
import argparse


def main(source, target):

    with (open(source, "r") as s,
          open(target, "r") as t,
          open("res_src", "w") as o_s,
          open("res_trg", "w") as o_t):
        
        src = s.readlines()
        trg = t.readlines()

        res = list(zip(src, trg))
        random.shuffle(res)

        for i, (src_sentence, trg_sentence) in enumerate(res):
            o_s.write(src_sentence)
            o_t.write(trg_sentence)

            if i % 500_000 == 0:
                print(f"Iteration: {i:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make pavlov corpora suitable for Fast Align")
    parser.add_argument("-s", help = "Path to source DNTs file")
    parser.add_argument("-t", help = "Path to target DNTs file")


    args = parser.parse_args()

    main(args.s, args.t)