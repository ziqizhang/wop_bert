import datetime

from sklearn.metrics import classification_report, accuracy_score
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels


def outputPredictions(pred, truth, model_name, task, outfolder):
    filename = os.path.join(outfolder, "predictions-%s-%s.csv" % (model_name, task))
    file = open(filename, "w")
    for p, t in zip(pred, truth):
        if p==t:
            line=str(p)+","+str(t)+",ok\n"
            file.write(line)
        else:
            line=str(p)+","+str(t)+",wrong\n"
            file.write(line)
    file.close()

def saveOutput(prediction, model_name, task,outfolder):
    filename = os.path.join(outfolder, "prediction-%s-%s.csv" % (model_name, task))
    file = open(filename, "w")
    for entry in prediction:
        file.write(str(entry)+"\n")
    file.close()


def prepare_score_string(p, r, f1, s, labels, target_names, digits):
    string = ",precision, recall, f1, support\n"
    for i, label in enumerate(labels):
        string= string+target_names[i]+","
        for v in (p[i], r[i], f1[i]):
            string = string+"{0:0.{1}f}".format(v, digits)+","
        string = string+"{0}".format(s[i])+"\n"
        #values += ["{0}".format(s[i])]
        #report += fmt % tuple(values)

    #average
    string+="mac avg,"
    for v in (np.average(p),
              np.average(r),
              np.average(f1)):
        string += "{0:0.{1}f}".format(v, digits)+","
    string += '{0}'.format(np.sum(s))
    return string

def save_scores(predictions, gs, model_name, target_and_feature,
                algorithm_param_identifier, digits, outfolder):
    outputPredictions(predictions, gs, model_name, target_and_feature, outfolder)
    filename = os.path.join(outfolder, "%s-%s.csv" % (model_name, target_and_feature))
    file = open(filename, "a+")
    file.write(algorithm_param_identifier)

    file.write("N-fold results:\n")
    labels = unique_labels(gs, predictions)
    target_names = ['%s' % l for l in labels]
    p, r, f1, s = precision_recall_fscore_support(gs, predictions,
                                                  labels=labels)
    acc=accuracy_score(gs, predictions)
    mac_prf_line=prepare_score_string(p,r,f1,s,labels,target_names,digits)

    prf_mac_weighted=precision_recall_fscore_support(gs, predictions,
                                                     average='weighted')
    line = mac_prf_line + "\nmacro avg weighted," + \
           str(prf_mac_weighted[0]) + "," + str(prf_mac_weighted[1]) + "," + \
           str(prf_mac_weighted[2]) + "," + str(prf_mac_weighted[3])

    prf = precision_recall_fscore_support(gs, predictions,
                                          average='micro')
    line=line+"\nmicro avg,"+str(prf[0])+","+str(prf[1])+","+\
         str(prf[2])+","+str(prf[3])
    file.write(line)
    file.write("\naccuracy on this run="+str(acc)+"\n\n")

    file.close()


def index_max(values):
    return max(range(len(values)), key=values.__getitem__)


def print_eval_report(best_params, cv_score, prediction_dev,
                      time_predict_dev,
                      time_train, y_test):
    print("CV score [%s]; best params: [%s]" %
          (cv_score, best_params))
    print("\nTraining time: %fs; "
          "Prediction time for 'dev': %fs;" %
          (time_train, time_predict_dev))
    print("\n %fs fold cross validation score:" % cv_score)
    print("\n test set result:")
    print("\n" + classification_report(y_test, prediction_dev))


def timestamped_print(msg):
    ts = str(datetime.datetime.now())
    print(ts + " :: " + msg)
