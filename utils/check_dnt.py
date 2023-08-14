#!/usr/bin/env python
# coding: utf-8

import argparse
import re
from collections import Counter

def list_intersection(b, a):
    return list((Counter(a) & Counter(b)).elements())

def main():

    parser = argparse.ArgumentParser(description='Check DNT matching between reference and hypothesis. Works line by line.')
    parser.add_argument('--ref', help='reference file', required=True)
    parser.add_argument('--hyp', help='hypothesis file', required=True)
    parser.add_argument('--src', help='optional, source language file', default=argparse.SUPPRESS)
    args = parser.parse_args()

    with open(args.ref, encoding='utf-8') as r:
        rlines = [line.strip() for line in r]
    with open(args.hyp, encoding='utf-8') as h:
        hlines = [line.strip() for line in h]
    if "src" in args:
        with open(args.src, encoding='utf-8') as s:
            slines = [line.strip() for line in s]

    rcount = 0
    hcount = 0
    mcount = 0
    for i, (rl, hl) in enumerate(zip(rlines, hlines)):
        rdnt = re.findall('\${DNT0}\s*\d+', rl, flags=re.I)
        hdnt = re.findall('\${DNT0}\s*\d+', hl, flags=re.I)
        rdnt = ["".join(elem.upper().split()) for elem in rdnt]
        hdnt = ["".join(elem.upper().split()) for elem in hdnt]
        rdnt.sort()
        hdnt.sort()
        rcount = rcount + len(rdnt)
        hcount = hcount + len(hdnt)
        mcount = mcount + len(list_intersection(rdnt, hdnt))
        if not(rdnt == hdnt):
            print()
            print("[Error] DNTs do not match at line", i)
            print("--> Ref DNTs:", rdnt)
            print("--> Hyp DNTs:", hdnt)
            if "src" in args:
                print("Src:", slines[i])
            print("Ref:", rl)
            print("Hyp:", hl)
            print("\n ========= \n")
    
    print("-- Summary --")
    print("DNT in ref:", rcount)
    print("DNT in hyp:", hcount)
    print("Matching DNTs:", mcount)
    print("Precision:", round(mcount/hcount, 3))
    print("Recall:", round(mcount/rcount, 3))

if __name__ == "__main__":
    main()




