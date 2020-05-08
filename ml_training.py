# -*- coding: utf-8 -*-
###############################################################################
# MODULE: training
# this is module to train TF convolutional classifiers
#
# Author:   William Kinsman
# Created:  10.10.2015
###############################################################################

import os
import re
import scipy
import platform
import wikipedia
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from text_tools import (
    preprocessing,
    extraction,
    vectorizer,
    vocab_tools,
    ml_visualizations,
)

# if platform.system()=='Linux':   import fasttext as fasttext # fasttext for linux
# if platform.system()=='Windows': import fastText as fasttext # fasttext for windows
from multiprocessing import cpu_count
import fasttext as fasttext


def convolutional_classifier(
    pos, neg, windowsize=100, step=50, vocab=None, test_percent=0.2, min_vars=0
):
    """
    trains a convolutional classifier from two strings
    @param pos: list of positive texts
    @param neg: list of negative texts (these will be balled together)
    @param windowsize: size of convolutional window
    @param step: step size of window (higher = faster training, lower quality)
    @param vocab: preprocessed vocabulary, else uses pos/neg to create one
    @param test_percent: decimal used to partition some portion to testing
    """
    # vocab can be derived or input
    if not vocab:
        vocab = vocab_tools.build_vocab(pos + "\n" + neg)
    # neg = '\n'.join(neg)
    pos = list(set(pos))
    neg = list(set(neg))
    ML_input = vectorizer.tb_vectorizer(pos + neg, vocab)
    ML_output = [1] * len(pos) + [0] * len(neg)

    # allocate some data to training if necessary
    if test_percent:
        pos_split = int(test_percent * len(pos))
        neg_split = int(test_percent * len(neg))
        pos_test = pos[:pos_split]
        neg_test = neg[:neg_split]
        pos = pos[pos_split:]
        neg = neg[neg_split:]

    # train the classifier
    classifier = SVC(C=0.1, kernel="linear", probability=True).fit(
        ML_input, ML_output
    )

    # if a percentage of test is requested
    if test_percent:

        # get results and actual
        y_true = [1] * pos_test.shape[0] + [0] * neg_test.shape[0]
        y_score = (
            classifier.predict_proba(
                vectorizer.tb_vectorizer(pos_test + neg_test, vocab)
            )[:, 1].tolist()
            + classifier.predict_proba(neg_test)[:, 1].tolist()
        )
        ml_visualizations.plot_performance(y_true, y_score)
        ml_visualizations.plot_coefficients(classifier, vocab)

    # return results
    if vocab:
        return classifier
    else:
        return (vocab, classifier)


def wiki_ensemble(
    articles,
    windowsize=100,
    step=5,
    vocab=None,
    balance=True,
    return_vocab=True,
):
    """
    train an ensemble of convolutional classifer from wikipedia
    @param windowsize: size of convolutional window
    @param step: step size of window (higher = faster training, lower quality)
    @param vocab: preprocessed vocabulary, else uses pos/neg to create one
    """
    # collect articles and preprocess
    regex_headers = re.compile(r"==(.*)==")
    articles = {
        k.lower(): re.sub(regex_headers, "", wikipedia.page(k).content)
        for k in articles
    }
    assert len(set(articles.values())) == len(
        articles
    ), "Two of the articles are the same. Aborting."
    ensemble = {
        k: {"classifier": None, "description": v[:30]}
        for k, v in articles.items()
    }

    # resolve vocab
    return_vocab = False
    if not vocab:
        return_vocab = True
        vocab = vocab_tools.build_vocab("\n".join(articles.values()))

    # preprocess and vectorize
    articles = {k: preprocessing.preprocess(v) for k, v in articles.items()}
    articles = {
        k: list(
            set(
                extraction.build_convolutions(
                    v, windowsize, step, metadata=False
                )
            )
        )
        for k, v in articles.items()
    }

    # use shortest length for balancing
    if balance:
        min_exemplars = min([len(i) for i in articles.values()])
        articles = {k: v[:min_exemplars] for k, v in articles.items()}

    # pre-vectorize
    print("vectorizing...")
    for a in tqdm(articles.keys()):
        articles[a] = vectorizer.tb_vectorizer(articles[a], vocab)

    # train each classifier
    print("training classifiers...")
    for a in tqdm(articles.keys()):
        pos = articles[a]
        neg = scipy.sparse.vstack([v for k, v in articles.items() if k != a])
        ML_input = scipy.sparse.vstack([pos, neg])
        ML_output = [1] * pos.shape[0] + [0] * neg.shape[0]
        ensemble[a]["classifier"] = SVC(
            C=0.01, kernel="linear", probability=True
        ).fit(ML_input, ML_output)

    # return results
    if return_vocab:
        return ensemble, vocab
    else:
        return ensemble


def fasttext_train(exemplars, labels, model_name="fasttext_model"):
    """
    @param exemplars: list of strings
    @param labels: list of labels for each string
    @param model_name: name of the model if preferred
    """
    # initialization
    assert len(exemplars) == len(labels)

    # insert all the training data into a text file
    #    with open('training.txt','a') as f:
    #        for i in range(len(exemplars)):
    #            f.write('__label__'+str(labels[i])+'  '+str(exemplars[i])+'\n')

    # train and drop model
    model = fasttext.train_supervised(
        "CRG Training/training_new_crg_0.05.txt",
        label="__label__",
        lr=0.1,
        dim=128,
        epoch=25,
        loss="hs",
        neg=200,
        wordNgrams=2,
        thread=cpu_count(),
        lrUpdateRate=150,
        verbose=2,
    )
    model.save_model(model_name + ".bin")


def fasttext_deploy(exemplars, model_name="fasttext_model", threshold=None):
    """
    @param exemplars: list of strings
    @param model_name: name of the model if preferred
    """

    # initialize
    if isinstance(exemplars, str):
        exemplars = [exemplars]
    model = fasttext.load_model(model_name + ".bin")

    # generate output depending on system
    output = model.predict(exemplars)  # 1 for probabilities
    output = [
        [output[0][i][0], output[1][i][0]] for i in range(len(output[0]))
    ]  # get first instance of each

    # remobe labels and return
    output = [[re.sub(r"^__label__", r"", i[0]), i[1]] for i in output]
    if threshold:
        output = [i if i[1] > threshold else ["", i[1]] for i in output]
    return output


def manually_review_data(data):
    """
    use this function to hand train text exemplars
    """
    good = []
    bad = []
    for i in range(len(data)):
        print("\n" + str(data[i]) + "\n")
        answer = input("is it good? [(y)es/(n)o/(c)ontinue/(e)nd]")
        if answer == "y":
            good.append(data[i])
        elif answer == "n":
            bad.append(data[i])
        elif answer == "c":
            continue
        elif answer == "e":
            return (good, bad)
        else:
            continue
    return (good, bad)
