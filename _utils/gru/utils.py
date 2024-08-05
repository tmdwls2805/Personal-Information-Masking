import random
import logging
import torch
import numpy as np
import pandas as pd


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_report_to_txt(config, accuracy, model, save_path, report):
    file = open(save_path+'result_{}_{}_{}.txt'.format(config.epochs, model, config.data_name), 'w')
    lines = report.split('\n')
    for line in lines:
        file.write(line)
        file.write('\n')
    file.write('accuracy        {}'.format(round(accuracy, 4)))
    file.close()


