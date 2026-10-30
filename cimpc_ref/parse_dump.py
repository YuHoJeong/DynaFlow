"""C++ dump_gradient 출력 파서."""
import numpy as np

def parse(path):
    blocks, cur, rows = {}, None, []
    for line in open(path):
        if line.startswith("### "):
            if cur: blocks[cur[0]] = np.array(rows, float).reshape(cur[1], cur[2])
            p = line.split(); cur = (p[1], int(p[2]), int(p[3])); rows = []
        elif cur is not None:
            v = line.split()
            if v and all(c in "0123456789+-.eE" for c in v[0]): rows.extend(float(x) for x in v)
    if cur: blocks[cur[0]] = np.array(rows, float).reshape(cur[1], cur[2])
    return blocks
