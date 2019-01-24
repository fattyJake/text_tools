# -*- coding: utf-8 -*-
###############################################################################
# Module:      visualizations
# Description: repo of tools for visualization
# Authors:     William Kinsman
# Created:     11.06.2017
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_curve
from text_tools import preprocessing,vectorizer,extraction

def plot_coefficients(classifier,variables,n=50,bottom=False):
    """
    @param classifier: name of the classifier to use in the pickleObj
    @param n: returns top/bottom n variables
    @param bottom: returns bottom n variables
    return: A plot of all or top n variables in the classifier
    """
    # initialize
    if not classifier: return None
    coefs = classifier.coef_
    if isinstance(coefs,np.ndarray): coefs = coefs.tolist()
    else: coefs = list(coefs.toarray()[0])
    if len(variables)<n: n=len(variables)
    assert len(coefs)==len(variables),"ERROR: variable-coefficient size mismatch. Aborting."
    
    # select top n
    combined = [i for i in zip(variables,coefs)]
    combined.sort(key=lambda x:x[1], reverse=True)
    variables = [i[0] for i in combined]
    coefs = [i[1] for i in combined]
    coefs,variables = (list(t) for t in zip(*sorted(zip(coefs,variables),reverse=True)))
    if bottom: 
        variables = variables[0:n] + variables[-n:]
        coefs = coefs[0:n] + coefs[-n:]
    else:
        variables = variables[0:n]
        coefs = coefs[0:n]    
    variables = variables[::-1]
    coefs = coefs[::-1]
    
    # plot
    spacing = 3
    barwidth = 0.2
    plt.figure(figsize=(5,((len(variables)+1) + 2*spacing)*barwidth))
    plt.barh(np.arange(len(variables)),coefs,align='center',alpha=0.5)
    plt.xlabel('Coefficients')
    plt.yticks(np.arange(len(variables)),variables)
    plt.ylim((-1*barwidth*4,len(variables)-barwidth))
    plt.grid(linestyle='dashed',axis='x')
    if bottom: plt.title('Top & Bottom ' + str(int(n)) + ' Variable Coefficients')
    else:      plt.title('Top ' + str(int(n)) + ' Variable Coefficients')
    plt.show()

def plot_confidences(text,ensemble,vocab,windowsize,step=5):
    """
    plot of confidences of each classifier in descending order
    @param text: plot the text in a document
    @param ensemble: ensemble
    @param varaibles: a list of the variables the classifier uses
    @param windowsize: character size of sliding window
    @param step: step size of sliding window
    """
    # initialize
    if isinstance(text,list): text = '\n'.join(text)
    text       = preprocessing.preprocess(text)
    windows    = extraction.build_convolutions(text,windowsize,step=step)
    vectors    = vectorizer.tb_vectorizer(windows,vocab)
    classifier_names = [i for i in ensemble.keys()]
    
    # evaluate patient for HCCs (if no classifier, P(HCC)=0)
    peak_prob = []
    for classifier in classifier_names:
        if ensemble[classifier]['classifier']: peak_prob.append(max(ensemble[classifier]['classifier'].predict_proba(vectors)[:,1]))
        else: peak_prob.append(0)
    peak_prob,classifier_names = (list(t) for t in zip(*sorted(zip(peak_prob,classifier_names))))
    
    # plot results
    spacing  = 3
    barwidth = 0.2
    plt.figure(figsize=(5,((np.sum(len(classifier_names))+1) + (2)*spacing)*barwidth))
    plt.barh(np.arange(len(classifier_names)),peak_prob,align='center',alpha=0.5)
    plt.title('Maximum Confidences of Detection of each Classifier')
    plt.xlabel('Maximum Confidence of Detection')
    plt.xlim((0,1))
    plt.ylim((-1*barwidth*4,len(classifier_names)-barwidth))
    plt.yticks(np.arange(len(classifier_names)),[ensemble[i]['description'] + ' ' +str(i) for i in classifier_names])
    plt.grid(linestyle='dashed',axis='x')
    plt.show()
    
def print_findings(text,classifier,variables,windowsize=100,step=50,threshold=0.5):
    """
    print the findings by a convolutional classifier
    @param text: plot the text in a document
    @param classifier: name of a classifier within the ensemble
    @param document: a list of strings representing pages
    """
    # build vectors
    if isinstance(text,list): text = '\n'.join(text)
    text       = preprocessing.preprocess(text)
    text        = preprocessing.preprocess(text,negex=True)
    pages,token_s,token_e,vectors = zip(*extraction.build_convolutions(text,windowsize,step=step,metadata=True))
    vectors    = vectorizer.tb_vectorizer(vectors,variables)
    
    # iterate over all classifiers
    if isinstance(classifier,dict):
        for i in classifier:
            if not classifier[i]['classifier']: continue
            prob = classifier[i]['classifier'].predict_proba(vectors)[:,1]
            if any(prob>threshold):
                print('\n'+str(i)+': '+str(classifier[i]['description']))
                for k in range(len(prob)):
                    if prob[k]>=threshold: print("Conf: "+format(prob[k],'.2f')+'\t'+vectors[k])
    
    # or just a single if provided
    else:
        prob = classifier.predict_proba(vectors)[:,1]
        for i in range(len(prob)):
            if prob[i]>=threshold: print("Conf: "+format(prob[i],'.2f')+'\t'+vectors[i])
    
def plot_performance(y_true,y_score,datapoint=False,title=None):
    """
    @param y_true: list of output booleans indicicating if True
    @param y_score: list of probabilities
    @param datapoint: (recall,precision) data point if provided
    do: plot ROC,PR,PT
    """
    # receiver operating characteristic
    plt.figure(1,figsize=(14,3))
    plt.subplot(131)
    fpr,tpr,_ = roc_curve(y_true,y_score)
    plt.plot(fpr,tpr, color='darkorange',lw=2,label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # precision recall
    plt.subplot(132)
    precision,recall,thresholds = precision_recall_curve(y_true, y_score)
    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    if datapoint: plt.plot(datapoint[0],datapoint[1],marker='x', markersize=5,color="red")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall')

    # precision threshold
    plt.subplot(133)
    precision,recall,thresholds = precision_recall_curve(y_true, y_score)
    plt.scatter(thresholds, precision[:-1], color='k',s=1)
    if datapoint: plt.plot((0,1),(datapoint[1],datapoint[1]),color="red")
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid()
    plt.title('Precision Threshold')

    if title: plt.suptitle(title)
    plt.show()
    
def plot_timeline(text,classifier,vocab,windowsize,step=5):
    """
    @param text: plot the text in a document
    @param classifier: classifier or ensemble
    @param windowsize: character size of sliding window
    @param step: step size of sliding window
    """
    # initialize
    if isinstance(text, list): text = '\n'.join(text)
    text       = preprocessing.preprocess(text)
    pages,word_s,word_e,windows = zip(*extraction.build_convolutions(text,windowsize,step=step,metadata=True))
    vectors    = vectorizer.tb_vectorizer(windows,vocab)
    
    # if its an ensemble plot for each
    if isinstance(classifier,dict):
        
        # get probabilities for each classifier
        for i in classifier:
            if not classifier[i]['classifier']: continue
            prob = classifier[i]['classifier'].predict_proba(vectors)[:,1]
            
            # plot results
            plt.figure(figsize=(12,3))
            plt.plot(np.arange(len(windows)),prob)
            plt.title('Confidence of Detection of \''+str(i)+'\' Across Document')
            plt.ylabel('Confidence of Detection')
            plt.xlabel('Token Index')
            plt.ylim((0,1))
            plt.xlim((-.5,len(windows)-.5))
            plt.grid(linestyle='dashed',axis='y')
            plt.show()
    else:
        # get probabilities
        prob = classifier.predict_proba(vectors)[:,1]
        
        # plot results
        plt.figure(figsize=(12,3))
        plt.plot(np.arange(len(windows)),prob)
        plt.title('Confidence of Detection of Classifier Across Document')
        plt.ylabel('Confidence of Detection')
        plt.xlabel('Token Index')
        plt.ylim((0,1))
        plt.xlim((-.5,len(windows)-.5))
        plt.grid(linestyle='dashed',axis='y')
        plt.show()