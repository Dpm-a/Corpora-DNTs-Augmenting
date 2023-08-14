import re
import sys
from collections import Counter

file1 = sys.argv[1]
file2 = sys.argv[2]
to_int = isinstance(sys.argv[-1],int) if len(sys.argv) > 2 else 0

#extract language
lng_src, lng_trg = re.findall(r'(train\.\w{2,3}-\w{2,3}\.\w{2,3})', file1 + file2)
print(f" ... Parsing [{lng_src}] and [{lng_trg}] ... \n")

dnt_count_f1 = 0
dnt_count_f2 = 0

with open(file1, "r") as f1, \
     open(file2, "r") as f2:
         
    counter_s = Counter()
    counter_t = Counter()

    dnt_count_f1, dnt_count_f2 = 0, 0
    pattern = r"\$\{DNT0\}(\d*)"

    for sent1, sent2 in zip(f1, f2):
        dnts_s = re.findall(pattern, sent1)
        dnts_t = re.findall(pattern, sent2)
        dnt_count_f1 += len(dnts_s)
        dnt_count_f2 += len(dnts_t)

        counter_s.update( list(map(lambda x: int(x),dnts_s)) if to_int else dnts_s)
        counter_t.update( list(map(lambda x: int(x),dnts_t)) if to_int else dnts_t)

    assert counter_s == counter_t, "DNTs don't match"
    print(*sorted(counter_s.items(), key= lambda x: x[0]), sep = "\n")

print()
print(f"{file1} dnts: {dnt_count_f1}")
print(f"{file2} dnts: {dnt_count_f2}")
print(f"Total DNTS: {dnt_count_f1 + dnt_count_f2}")
