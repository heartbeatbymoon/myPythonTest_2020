#encoding=utf-8
import os
import code
# import config
import numpy as np
import pickle
import utils

def main(args):

    train_en,train_cn = utils.load_data(args.trans_file)
    code.interact(local=locals())