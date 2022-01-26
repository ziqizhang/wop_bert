import datetime
import textwrap

import pandas as pd
import numpy, os, gzip, json

def get_next_batch(input_folder, batch_size, start_from_file=None, start_from_batch=None):
    #look into the input folder and find the starting file
    onlyfiles = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    onlyfiles.sort()

    found_start_file=True
    if start_from_file is not None and start_from_file!='None':
        found_start_file=False

    for f in onlyfiles:
        if not found_start_file:
            if str(f).lower()==start_from_file.lower():
                found_start_file = True
            else:
                continue

        print("Current processing file {} time={} ".format(f, datetime.datetime.now()))
        #now work out batch
        path = os.path.join(input_folder, f)
        if start_from_batch is None or start_from_batch=='None':
            start_from_batch=0

        startindex = batch_size*int(start_from_batch)
        currentbatch=start_from_batch
        currentbatchsize=0
        index=0


        data=[]
        for j in parse_gzip(path):
            if index<startindex:
                index+=1
                continue
                # category, rating, reviewText, label, vote, verified, reviewerID, asin, summary
            rating = 0
            if 'overall' in j.keys():
                rating=j['overall']
            text = ''
            if 'reviewText' in j.keys():
                text = j['reviewText']
                text=textwrap.shorten(text, width=3000)
            vote = 0
            if 'vote' in j.keys():
                vote = j['vote']
            verified = False
            if 'verified' in j.keys():
                verified = j['verified']
            reviewerID = ""
            if 'reviewerID' in j.keys():
                reviewerID = j['reviewerID']
            asin = ""
            if 'asin' in j.keys():
                asin = j['asin']
            summary = ""
            if 'summary' in j.keys():
                summary = j['summary']
            row=[str(f), rating, text,'CG', vote,
                 verified, reviewerID, asin, summary]
            data.append(row)

            currentbatchsize+=1
            if currentbatchsize==batch_size:
                #print("\t generating batch: {}, size={}".format(currentbatch, len(data)))
                yield numpy.array(data), currentbatch, str(f)
                currentbatchsize=0
                data.clear()
                currentbatch+=1
        #print("\t final batch: {}, size={}".format(currentbatch, len(data)))
        yield numpy.array(data), currentbatch, str(f)
        data.clear()



def parse_gzip(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)


def get_next_batch_test(input_folder, batch_size, start_from_file=None, start_from_batch=None):
    data = load_and_merge_train_test_data_productfakerev(input_folder+"/fakeproductrev_train.csv")
    return data, 0


def load_and_merge_train_test_data_productfakerev(test_data_file):
    test = pd.read_csv(test_data_file, header=0, delimiter=",", quoting=0, encoding="utf-8",
                       ).fillna('').values
    for row in test:
        row[3]='CG'
    test.astype(str)

    return test

if __name__ == "__main__":
    input_folder="/home/zz/Work/data/amazon/all/part"
    batchsize=5000
    start_from_file=None
    start_from_batch=None
    #category, rating, reviewText, label, vote, verified, reviewerID, asin, summary
    print(datetime.datetime.now())
    for batch, id, source in get_next_batch(input_folder, batchsize, start_from_file, start_from_batch):
        print("{}, id={}, cat={} prod={}".format(len(batch), id, batch[0][0], batch[0][7]))
    print(datetime.datetime.now())