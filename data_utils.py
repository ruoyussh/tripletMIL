from torch.utils.data import Dataset
import random
import numpy as np
import os, pickle


class MyDataset(Dataset):
    def __init__(self, bags, isTest=False, sample_max_num=-1):
        self.bags = bags
        self.isTest = isTest
        self.sample_max_num = sample_max_num

    def __getitem__(self, index):
        examples = self.bags[index]
        if not self.isTest and self.sample_max_num > 0:
            examples = list(examples)
            random.shuffle(examples)
            examples = examples[:self.sample_max_num]
            examples = np.array(examples)
        return examples

    def __len__(self):
        return len(self.bags)


def read_features(patient_list, patient_labels, fold_pt_dir):
    bags = []
    labels = []

    for idx, patient_id in enumerate(patient_list):
        label = patient_labels[patient_id]
        pkl_name = fold_pt_dir + patient_id + '.pkl'
        if not os.path.exists(pkl_name):
            print(pkl_name, 'does not exist!!!')
            continue
        with open(pkl_name, 'rb') as handle:
            pt_file = pickle.load(handle)

        pt_bags = pt_file

        bags.append(np.array(pt_bags))
        labels.append(label)
    bags = np.array(bags)
    labels = np.array(labels)
    return bags, labels