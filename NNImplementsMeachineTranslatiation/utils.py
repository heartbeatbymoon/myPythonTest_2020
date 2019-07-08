# 写一些辅助性的func
import io

def load_data(in_file):
    en = []
    cn = []
    # encoding = "utf-8"不加这个会报错
    with io.open(in_file, "r",encoding="utf-8") as f:
        for line in f:
            # strip除掉首尾垃圾信息
            line = line.strip().split("\t")
            en.append(line[0])
            cn.append(line[1])
    return en, cn
