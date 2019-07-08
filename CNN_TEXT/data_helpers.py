# encoding=utf-8
import numpy as np

def load_data_and_labels(positive_data_file, negative_data_file):
    positive = open(positive_data_file,"rb").read().decode("utf-8")
    negative = open(negative_data_file,"rb").read().decode("utf-8")

    positive_examples =positive.split("\n")[:-1]
    negative_examples =negative.split("\n")[:-1]

    positive_examples=[s.strip() for s in positive_examples]
    negative_examples=[s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    positive_label = [[0,1] for _ in positive_examples]
    negative_label = [[0,1] for _ in negative_examples]
    y = np.concatenate([positive_label,negative_label])