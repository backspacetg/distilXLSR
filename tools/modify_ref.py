import re
import sys

hypo_file = sys.argv[1]
ref_file = sys.argv[2]
ref_output = sys.argv[3]

with open(ref_file) as in_f:
    refs = in_f.readlines()


find_number = re.compile(r'\(None-(.*)\)')
with open(hypo_file) as f, open(ref_output, 'w') as f_out:
    line = f.readline()
    while line:
        rank = find_number.search(line)
        rank = (int(rank.group(1)))
        real_ref = refs[rank].strip()
        f_out.write("{} (None-{})\n".format(real_ref, rank))
        line = f.readline()
