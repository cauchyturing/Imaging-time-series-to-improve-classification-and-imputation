# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:45:33 2014
Modified on Wed Jan 27 15:36:00 2016
@author: Stephen Wang
"""

import pandas as pd
import numpy as np
import scipy.io as sio
import pickle

def output_arff(dataset, filename="data.arff"):

    outfile = open(filename, 'w')

    features = {}
    labels = {}
    
    # Find all unique features and class labels
    for datum in dataset:
        label = datum[0]
        bop = datum[1]

        labels[label] = 1
        for key in bop.keys():
            features[key] = 1

    # Stores features and labels as sorted lists
    features = sorted(features.keys())
    labels = sorted(labels.keys())

    # Write relation
    outfile.write('@RELATION QM\n\n')

    # Write feature names and types
    for feature in features:
        outfile.write('@ATTRIBUTE ' + feature + ' NUMERIC\n')

    # Write class labels
    outfile.write('@ATTRIBUTE class {')
    for label in labels:
        outfile.write(str(label))
        if not label == labels[-1]:
            outfile.write(',')
    outfile.write('}\n\n')

    # Write data instances
    outfile.write('@DATA\n')
    for datum in dataset:
        label = datum[0]
        bop = datum[1]

        for feature in features:
            if feature in bop:
                outfile.write(str(bop[feature]))
            else:
                outfile.write('0')
            outfile.write(',')

        outfile.write(str(label))
        outfile.write('\n')

    outfile.close()
    
def QV(series, Q):
    q = pd.qcut(series, Q)
    qv = np.zeros([1,Q])
    for i in range(0, q.labels.size):
        qv[0][q.labels[i]] += 1
    return np.array(qv[0][:]/sum(qv[0][:]))
    
def QVeq(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    qv = np.zeros([1,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0,len(label)):
        qv[0][label[i]] += 1.0        
    return np.array(qv[0][:]/sum(qv[0][:]))
    
def QM(series, Q):
    q = pd.qcut(series, Q)
    MSM = np.zeros([Q,Q])
    for i in range(0, q.labels.size-1):
        MSM[q.labels[i]][q.labels[i+1]] += 1
    for i in range(Q):
        MSM[i][:] = MSM[i][:]/sum(MSM[i][:])
    return np.array(MSM)
    
def QMeq(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    MSM = np.zeros([Q,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0, len(label)-1):
        MSM[label[i]][label[i+1]] += 1
    for i in range(Q):
        if sum(MSM[i][:]) == 0:
            continue
        MSM[i][:] = MSM[i][:]/sum(MSM[i][:])
    return np.array(MSM)

def QMeq_smooth(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    MSM = np.zeros([Q,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0, len(label)-1):
        MSM[label[i]][label[i+1]] += 1
    for i in range(Q):
        if sum(MSM[i][:]) == 0:
            MSM[i][:] = 1.0/Q
            continue
        MSM[i][:] = MSM[i][:]/sum(MSM[i][:])
    return np.array(MSM)
     
def writeQM(qm, path):    
    size = len(qm)
    f = open(path,'wb')
    for i in range(size):
        f.write(';'+str(i))
    f.write('\n')
    for i in range(size):
        f.write(str(i)+';'+';'.join(map(str, list(qm[i][:].A1))[1:-1])+'\n')
    f.close()

def writeQM2arff(qm, label, path):
    arff = []
    for k in range(len(qm)):
        size = len(qm[k])
        dic = {}
        for i in range(0,size):
            for j in range(0, size):
                dic[str(i)+'_'+str(j)] = str(qm[k][i,j])
        arff.append([label[k], dic])
    
    output_arff(arff, path)
    
def writeQM2pkl(qm, label, path, tr, te, va):
    neg = []
    pos = []
    negl = []
    posl = []
    label = map(float, label)
    if len(qm) != len(label):
        print("Length of QM and length of labels are not matching!")
    for k in range(len(qm)):
        if label[k] == 1.0:
            pos.append(qm[k].A1)
            posl.append(label[k])
        else:
            neg.append(qm[k].A1)
            negl.append(label[k])
#            negl.append(-1.0)
    traindata = np.array(pos[:int(len(pos)*tr)]+neg[:int(len(neg)*tr)], \
        dtype = 'float32')
    testdata = np.array(pos[int(len(pos)*tr):int(len(pos)*(tr+te))]+neg[int(len(neg)*tr):int(len(neg)*(tr+te))], \
        dtype = 'float32')
    validatedata = np.array(pos[int(len(pos)*(tr+te)):]+neg[int(len(neg)*(tr+te)):], \
        dtype = 'float32')

    trainlabel = np.array(posl[:int(len(posl)*tr)]+negl[:int(len(negl)*tr)], \
        dtype = 'float32')
    testlabel = np.array(posl[int(len(posl)*tr):int(len(posl)*(tr+te))]+negl[int(len(negl)*tr):int(len(negl)*(tr+te))], \
        dtype = 'float32')
    validatelabel = np.array(posl[int(len(posl)*(tr+te)):]+negl[int(len(negl)*(tr+te)):], \
        dtype = 'float32')
        
    train = (traindata, trainlabel)
    test = (testdata, testlabel)
    validate = (validatedata, validatelabel)
    
    pkldata = (train, test, validate)
    with open(path, 'wb') as pklfile:
        pickle.dump(pkldata, pklfile, protocol = pickle.HIGHEST_PROTOCOL)

#def jordan(qm):
#    size = len(qm)
#    (P,J) = sympy.Matrix(list(qm)).jordan_form()
#    return np.matrix(list(P),dtype = 'float32').reshape(size, size), np.matrix(list(J),dtype = 'float32').reshape(size, size)     

def writeQM2PretrainMat(name, QM, s):
    sio.savemat(name+'_pretrain_data.mat',{'X':QM,'input_ch':s, 'datastr':name+'_pretrain'})

def writeQM2Mat(name, QM, label):    
    sio.savemat(name+'_data.mat',{'X':QM,'Y':label})
#k = 0
#fn = 'D:\Dropbox\Time_Series\UCRdata\dataset\coffee\coffee_MYTEST'
#data = open(fn).readlines()
#data = [map(float, each.strip().split()) for each in data]
#plt.figure();plt.imshow(QM(data[k][1:], 10));plt.title('label'+str(data[k][0]))
#print len(data), data[k][0]        
