from exp import exp_util
import sys
import torch
from classifier import classifier_bert_
from wordsegment import load
load()

if __name__ == "__main__":
    # argv-1: folder containing all settings to run, see 'input' folder
    # argv-2: working directory
    # argv3,4:set to False

    # the program can take additional parameters to overwrite existing ones defined in setting files.
    # for example, if you want to overwrite the embedding file, you can include this as an overwrite
    # param in the command line, but specifying [embedding_file= ...] where 'embedding_file'
    # must match the parameter name. Note that this will apply to ALL settings
    overwrite_params = exp_util.parse_overwrite_params(sys.argv)

    mwpd_fieldname_to_colindex_map = {
        'ID': 0,
        'Name': 1,
        'Description': 2,
        'CategoryText': 3,
        'URL': 4,
        'lvl1': 5,
        'lvl2': 6,
        'lvl3': 7,
    }
    ##    ID, Name, Desc, Brand, Manufacturer, URL, lvl1
    wdc_fieldname_to_colindex_map = {
        'ID': 0,
        'Name': 1,
        'Desc': 2,
        'Brand': 3,
        'Manufacturer': 4,
        'URL': 5,
        'lvl1': 6
    }

    icecat_fieldname_to_colindex_map = {
        'ID': 0,
        'Description.URL': 1,
        'Brand': 2,
        'SummaryDescription.LongSummaryDescription': 3,
        'Title': 4,
        'Category.CategoryID': 5,
        'Category.Name.Value': 6
    }

    rakuten_fieldname_to_colindex_map = {
        'Name': 0,
        'lvl1': 1
    }

    train = sys.argv[1]
    test = sys.argv[2]
    outfolder = sys.argv[3]

    if sys.argv[5] == 'mwpd':
        text_field_mapping = mwpd_fieldname_to_colindex_map
    elif sys.argv[5] == 'wdc':
        text_field_mapping = wdc_fieldname_to_colindex_map
    elif sys.argv[5] == 'rakuten':
        text_field_mapping = rakuten_fieldname_to_colindex_map
    else:
        text_field_mapping = icecat_fieldname_to_colindex_map

    setting_file = sys.argv[4]
    bert_model = sys.argv[6]

    properties = exp_util.load_properties(setting_file)

    print("loading dataset...")
    if sys.argv[5] == "mwpd":
        df_all, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonMPWD(train, test)
    elif sys.argv[5] == "rakuten":
        df_all, train_size, test_size = exp_util. \
            load_and_merge_train_test_csvRakuten(train, test, delimiter="\t")
    elif sys.argv[5] == "icecat":
        df_all, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonIceCAT(train, test)
    else:  # wdc
        df_all, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonWDC(train, test)
    print("data loaded")

    class_fieldname = exp_util.load_setting("class_fieldname", properties, overwrite_params)
    param_label_field = text_field_mapping[class_fieldname]
    param_sent_length = int(exp_util.load_setting("param_sentence_length", properties,
                                                  overwrite_params))
    param_batch_size = int(exp_util.load_setting("param_batch_size", properties,
                                                 overwrite_params))
    param_epoch = int(exp_util.load_setting("param_training_epoch", properties,
                                            overwrite_params))

    if bert_model.startswith("/"):
        bert_model=bert_model[bert_model.rfind("/")+1:]
    model_name = bert_model

    target_and_feature=setting_file[setting_file.rfind("/") + 1:]
    input_text_fields = []
    count = 0
    for x in exp_util.load_setting("text_fieldnames", properties, overwrite_params).split("|"):
        input_text_fields.append(text_field_mapping[x])

    classifier_bert_.fit_bert_holdout(df_all,
                                      train_size,
                                      param_label_field,
                                      param_sent_length,
                                      param_batch_size,
                                      param_epoch,
                                      bert_model,
                                      outfolder,
                                      model_name,
                                      target_and_feature,
                                      1,
                                      input_text_fields)


'''
"/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/small_train.json"
"/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/small_train.json"
/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020
/home/zz/Work/wop_bert/input/dnn_holdout/mwpd/n/gslvl1_name.txt
mwpd
bert-base-uncased
'''

'''
"/home/zz/Work/data/Rakuten/original/rdc-catalog-gold.tsv"
"/home/zz/Work/data/Rakuten/original/rdc-catalog-gold.tsv"
/home/zz/Work/data/Rakuten/original
/home/zz/Work/wop_bert/input/dnn_holdout/rakuten/n/gslvl1_name.txt
rakuten
bert-base-uncased
'''

'''
"/home/zz/Work/data/IceCAT/icecat_data_train.json"
"/home/zz/Work/data/IceCAT/icecat_data_test.json"
/home/zz/Work/data/IceCAT
/home/zz/Work/wop_bert/input/dnn_holdout/rakuten/n/gslvl1_name.txt
icecat
bert-base-uncased
'''