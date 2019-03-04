# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 22:18:57 2014
Modified on Wed Jan 27 15:36:00 2016
@author: Stephen Wang
"""

from serie2QMlib import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Define sliding window
def window_time_series(series, n, step = 1):
#    print "in window_time_series",series
    if step < 1.0:
        step = max(int(step * n), 1)
    return [series[i:i+n] for i in range(0, len(series) - n + 1, step)]

#PAA function
def paa(series, now, opw):
    if now == None:
        now = int(len(series) / opw)
    if opw == None:
        opw = int(len(series) / now)
    return [sum(series[i * opw : (i + 1) * opw]) / float(opw) for i in range(now)]

def standardize(serie):
    dev = np.sqrt(np.var(serie))
    mean = np.mean(serie)
    return [(each-mean)/dev for each in serie]

#Rescale data into [0,1]
def rescale(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap for each in serie]
#Rescale data into [-1,1]
def rescaleminus(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap*2-1 for each in serie]

#Generate quantile bins
def QMeq(series, Q):
    q_labels = pd.qcut(list(series), Q, labels=False)
    q_levels = pd.qcut(list(series), Q, labels=None)
    dic = dict(zip(series, q_labels))
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
    return np.array(MSM), q_labels, q_levels.categories.get_values().tolist()

#Generate quantile bins when equal values exist in the array (slower than QMeq)
def QVeq(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    qv = np.zeros([1,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0,len(label)):
        qv[0][label[i]] += 1.0
    return np.array(qv[0][:]/sum(qv[0][:])), label

#Generate Markov Matrix given a spesicif number of quantile bins
def paaMarkovMatrix(paalist,levels):
    paaindex = []
    for each in paalist:
        for level in levels:
            if each >=level.left and each <= level.right:
                paaindex.append(k)
    return paaindex

# Generate pdf files of generated images
def gengrampdfs(image,paaimages,label,name):
    import matplotlib.backends.backend_pdf as bpdf
    import operator
    index = zip(range(len(label)),label)
    index.sort(key = operator.itemgetter(1))
    with bpdf.PdfPages(name) as pdf:
        count = 0
        for p,q in index:
            count += 1
            print('generate fig of pdfs:'.format(p))
            plt.ioff();fig= plt.figure();plt.suptitle(datafile+'_'+str(label[p]));ax1 = plt.subplot(121);plt.imshow(image[p]);divider = make_axes_locatable(ax1);cax = divider.append_axes("right", size="5%", pad=0.1);plt.colorbar(cax = cax);ax2 = plt.subplot(122);plt.imshow(paaimage[p]);divider = make_axes_locatable(ax2);cax = divider.append_axes("right", size="5%", pad=0.1);plt.colorbar(cax = cax);
            pdf.savefig(fig)
            plt.close(fig)
            if count > 30:
                break
    pdf.close

# Generate pdf files of trainsisted array in porlar coordinates
def genpolarpdfs(raw,label,name):
    import matplotlib.backends.backend_pdf as bpdf
    import operator
    index = zip(range(len(label)),label)
    index.sort(key = operator.itemgetter(1))
    len(raw[0]) - 1
    with bpdf.PdfPages(name) as pdf:
        for p,q in index:
            print('generate fig of pdfs:'.format(p))
            plt.ioff();r = np.array(range(1,length+1));r=r/100.0;theta = np.arccos(np.array(rescaleminus(standardize(raw[p][1:]))))*2;fig=plt.figure();plt.suptitle(datafile+'_'+str(label[p]));ax = plt.subplot(111, polar=True);ax.plot(theta, r, color='r', linewidth=3);
            pdf.savefig(fig)
            plt.close(fig)
    pdf.close

#return the max value instead of mean value in PAAs
def maxsample(mat, s):
    retval = []
    x, y, z = mat.shape
    l = np.int(np.floor(y/float(s)))
    for each in mat:
        block = []
        for i in range(s):
            block.append([np.max(each[i*l:(i+1)*l,j*l:(j+1)*l]) for j in range(s)])
        retval.append(np.asarray(block))
    return np.asarray(retval)

#Pickle the data and save in the pkl file
def pickledata(mat, label, train, name):
    print('..pickling data:'.format(name))
    traintp = (mat[:train], label[:train])
    testtp = (mat[train:], label[train:])
    with open(name+'.pkl', 'wb') as f:
        pickletp = [traintp, testtp]
        pickle.dump(pickletp, f, protocol=pickle.HIGHEST_PROTOCOL)

def pickle3data(mat, label, train, name):
    print('..pickling data:'.format(name))
    traintp = (mat[:train], label[:train])
    validtp = (mat[:train], label[:train])
    testtp = (mat[train:], label[train:])
    with open (name+'.pkl', 'wb') as f:
        pickletp = [traintp, validtp, testtp]
        pickle.dump(pickletp, f, protocol=pickle.HIGHEST_PROTOCOL)

#################################
###Define the parameters here####
#################################

datafiles = ['Coffee_ALL'] # Data fine name
trains = [28] # Number of training instances (because we assume training and test data are mixed in one file)
size = [64]  # PAA size
quantile = [16] # Quantile size
reduction_type = 'patch' # Reduce the image size using: full, patch, paa


for datafile, train in zip(datafiles,trains):
    fn = datafile
    for s in size:  
        for Q in quantile:
            print('read file: {}, size: {}, reduction_type: {}'.format(datafile, s, reduction_type))
            raw = open(fn).readlines()
            raw = [list(map(float, each.strip().split())) for each in raw]
            length = len(raw[0])-1
            
            print('format data')
            label = []
            paaimage = []
            paamatrix = []
            patchimage = []
            patchmatrix = []
            fullimage = []
            fullmatrix = []
            for each in raw:
                label.append(each[0])
                #std_data = rescaleminus(each[1:])
                #std_data = rescale(each[1:])
                std_data = each[1:]
                #std_data = standardize(each[1:])
                #std_data = rescaleminus(std_data)
                
                paalist = paa(std_data,s,None) 
                
                ############### Markov Matrix #######################
                mat, matindex, level = QMeq(std_data, Q)
                ##paamat,paamatindex = QMeq(paalist,Q)
                paamatindex = paaMarkovMatrix(paalist, level)
                column = []
                paacolumn = []
                for p in range(len(std_data)):
                    for q in range(len(std_data)):
                        column.append(mat[matindex[p]][matindex[(q)]])
                        
                for p in range(s):
                    for q in range(s):
                        paacolumn.append(mat[paamatindex[p]][paamatindex[(q)]])
                        
                column = np.array(column)
                columnmatrix = column.reshape(len(std_data),len(std_data))
                fullmatrix.append(column)
                paacolumn = np.array(paacolumn)
                paamatrix.append(paacolumn)
                
                fullimage.append(column.reshape(len(std_data),len(std_data)))
                paaimage.append(paacolumn.reshape(s,s))
                
                batch = int(len(std_data)/s)
                patch = []
                for p in range(s):
                    for q in range(s):
                        patch.append(np.mean(columnmatrix[p*batch:(p+1)*batch,q*batch:(q+1)*batch]))
                patchimage.append(np.array(patch).reshape(s,s))
                patchmatrix.append(np.array(patch))
 
            paaimage = np.asarray(paaimage)
            paamatrix = np.asarray(paamatrix)
            patchimage = np.asarray(patchimage)
            patchmatrix = np.asarray(patchmatrix)
            fullimage = np.asarray(fullimage)
            fullmatrix = np.asarray(fullmatrix)
            label = np.array(label)
            
            if reduction_type == 'patch':
                savematrix = patchmatrix
            elif reduction_type == 'paa':
                savematrix = paamatrix
            else:
                savematrix = fullmatrix
                
            datafilename = datafile +'_'+reduction_type+'_PAA_'+str(s)+'_Q_'+str(Q)+'_MTF'
            pickledata(savematrix, label, train, datafilename)

k=0;plt.figure();
plt.suptitle(datafile+'_index_'+str(k)+'_label_'+str(label[k])+'_Q_'+str(Q)+'_S_'+str(s));

ax1 = plt.subplot(121);plt.imshow(fullimage[k]);
plt.title('full image');
divider = make_axes_locatable(ax1);
cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);

# ax2 = plt.subplot(132);plt.imshow(paaimage[k]);plt.title('PAA image');
# divider = make_axes_locatable(ax2);cax = divider.append_axes("right", size="5%", pad=0.2);
# plt.colorbar(cax = cax);

ax3 = plt.subplot(122);
plt.imshow(patchimage[k]);
plt.title('patch average');
divider = make_axes_locatable(ax3);
cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);