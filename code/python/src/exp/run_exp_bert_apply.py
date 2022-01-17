from exp import exp_util
import sys
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

    fakerev_fieldname_to_colindex_map = {
        'Name': 2,
        'label': 3
    }

    task=sys.argv[1]
    setting_file = sys.argv[2]
    if sys.argv[3] == 'fakerev':
        text_field_mapping = fakerev_fieldname_to_colindex_map
    else:
        print("dataset '{}' not supported, exit".format(sys.argv[6]))
        exit(1)
    bert_model = sys.argv[4]
    if bert_model.startswith("/"):
        model_name = bert_model[bert_model.rfind("/") + 1:]
    else:
        model_name = bert_model

    properties = exp_util.load_properties(setting_file)
    class_fieldname = exp_util.load_setting("class_fieldname", properties, overwrite_params)
    param_label_field = text_field_mapping[class_fieldname]
    param_sent_length = int(exp_util.load_setting("param_sentence_length", properties,
                                                  overwrite_params))
    param_batch_size = int(exp_util.load_setting("param_batch_size", properties,
                                                 overwrite_params))
    param_epoch = int(exp_util.load_setting("param_training_epoch", properties,
                                            overwrite_params))
    target_and_feature = setting_file[setting_file.rfind("/") + 1:]
    input_text_fields = []
    count = 0
    for x in exp_util.load_setting("text_fieldnames", properties, overwrite_params).split("|"):
        input_text_fields.append(text_field_mapping[x])

    if task.lower()=='train':
        train = sys.argv[5]
        test = sys.argv[6]
        outfolder = sys.argv[7]

        print("loading dataset...")
        if sys.argv[3]=="fakerev":
            df_all, train_size, test_size = exp_util. \
                load_and_merge_train_test_data_productfakerev(train, test)

        print("data loaded")

        classifier_bert_.fit_bert_trainonly(df_all,
                                          train_size+test_size, #we use all data for training
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

    else:
        '''
        folder_to_classificationmodel,
               folder_to_bert_model: str,  # either a name identifying the pre-trained model, or path
               outfolder: str,
               param_label_field: int,
               param_sentence_length: int,
               param_batch_size: int,
               text_norm_option: int,
               text_input_fields: list,
               input_data_folder: str,
               input_data_batch_size: int,
               input_data_startfromfile=None,
               input_data_startfrombatch=None
        '''
        classifier_bert_.apply_mode(
            sys.argv[5], #classifier model, saved
            bert_model,
            sys.argv[6], #outfolder
                                            param_label_field,
                                            param_sent_length,
                                            param_batch_size,
            text_norm_option=1,
            text_input_fields=input_text_fields,
            input_data_folder=sys.argv[7],
            input_data_batch_size=10000,
            input_data_startfromfile=sys.argv[8],
            input_data_startfrombatch=sys.argv[9]
            )

#/home/li1zz/wop_bert/input/dnn_holdout/fakeprodrev/n
#/home/li1zz/data/fakeproductrev
