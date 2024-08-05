import os
from collections import Counter
#from bilm import dump_weights as dump_elmo_weights

def construct_elmo_vocab(corpus_fname, output_fname):
    #make_save_path(output_fname)
    count = Counter()
    num = 0
    with open(corpus_fname, 'r', encoding='cp949') as f1:
        for sentence in f1:
            tokens = sentence.replace('\n', '').strip().split(" ")
            for token in tokens:
                count[token] += 1
                num = num+1

    with open(output_fname, 'w', encoding='cp949') as f2:
        f2.writelines("</S>\n")
        f2.writelines("<S>\n")
        f2.writelines("<UNK>\n")
        #print("word 개수:", count.__len__())
        #print("word:", count)
        for word, _ in count.most_common(100000):
            f2.writelines(word + "\n")
    return num

def make_save_path(full_path):
    model_path = '/'.join(full_path.split("/")[:-1])
    print(model_path)
    if not os.path.exists(model_path):
        print(model_path)
        os.makedirs(model_path)


