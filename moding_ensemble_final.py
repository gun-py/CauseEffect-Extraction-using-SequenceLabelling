import sys
import numpy as np
import pandas as pd
from statistics import mode

def read_file(filename):
    text, tags = [], []
    with open(filename, encoding='utf-8') as f:
        t, ta = [], []
        for line in f.readlines():
            if line=="\n" and t and ta:
                text.append(t)
                tags.append(ta)
                t, ta = [], []
            else:
                splits = line.split()
                assert len(splits) <= 3
                t.append(splits[0])
                ta.append(splits[-1])
                
    print('-----------------------------------------------')
    print(filename, ':', len(text))
    return text, tags


text1, preds1 = read_file('/content/test_predictions1.txt')
_, preds2 = read_file('/content/test_predictions2.txt')
_, preds3 = read_file('/content/test_predictions3.txt')
_, preds4 = read_file('/content/test_predictions4.txt')
_, preds5 = read_file('/content/test_predictions5.txt')
_, preds6 = read_file('/content/test_predictions6.txt')
labels = {"_":0, "B-C":1, "I-C":2, "B-E":3, "I-E":4}
targets = ["_", "B-C", "I-C", "B-E", "I-E"]


ensemb_pred = []
count = 0
for j in range(len(preds1)):
    temp_pred = []
    for i in range(len(preds1[j])):
        try:
            temp_pred.append(mode([preds1[j][i], preds2[j][i], preds3[j][i], preds4[j][i], preds5[j][i], preds6[j][i]]))
        except: 
            print([preds1[j][i], preds2[j][i], preds3[j][i], preds4[j][i], preds5[j][i], preds6[j][i]])
            temp_pred.append(preds1[j][i])
            count += 1
    ensemb_pred.append(temp_pred)

for i in range(len(ensemb_pred)):
    if len(preds1[i]) != len(ensemb_pred[i]):
        print(i)
        print(preds1[i])
        print(preds2[i])
        print('-------------')


for i in range(len(preds1[6])):
    print(i, ':::::', [preds1[6][i], preds2[6][i], preds3[6][i], preds4[6][i]])

preds = post_process_bio(ensemb_pred)
sub = pd.read_csv('/content/task2.csv', delimiter=';')
print('Length of eval data:', len(sub))
cause, effect = submit(sub, preds, text1)
sub['Cause'] = cause
sub['Effect'] = effect
sub.to_csv('pred.csv', ';', index=0)