# python3 check.py chatgpt_text2shape2.csv 4096

import csv, sys

with open(sys.argv[1], 'r') as ifile:
    reader = csv.reader(ifile, delimiter=',', quotechar='"')
    _ = next(reader)
    b=0
    while True:
        l = []
        b+=1
        try:
            for i in range(int(sys.argv[2])):
                row = next(reader)
                l.append(row[1])
            try:
                assert(len(l)==len(set(l)))
            except AssertionError:
                print(f'Batch {b} wrong')
                print('  duplicates:', set([_ for _ in l if l.count(_)!=1]))
        except StopIteration:
            break