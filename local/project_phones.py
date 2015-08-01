#!/usr/bin/env python

import sys

phone_set = set()
f = open(sys.argv[1])
for line in f:
    parts = line.split()
    phone_set.add(parts[0])
f.close()

for line in sys.stdin:
    parts = line.split()

    if len(parts) == 5:
        if parts[2] in phone_set:
            print(line.strip())
    else:
        print(line.strip())
