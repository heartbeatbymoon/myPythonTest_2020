#encoding=utf-8
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    #data
    parser.add_argument('--train_file',type=str,default=None,
                        help="training file")


    return parser.parse_args()