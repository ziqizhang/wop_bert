import pandas as pd
import numpy

def get_next_batch(input_folder, batch_size, start_from_file=None, start_from_batch=None):
    data = load_and_merge_train_test_data_productfakerev(input_folder+"/fakeproductrev_train.csv")
    return data, 0


def load_and_merge_train_test_data_productfakerev(test_data_file):
    test = pd.read_csv(test_data_file, header=0, delimiter=",", quoting=0, encoding="utf-8",
                       ).fillna('').values
    for row in test:
        row[3]='CG'
    test.astype(str)

    return test