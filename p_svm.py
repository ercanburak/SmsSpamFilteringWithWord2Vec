from nltk import word_tokenize
import time
import numpy as np
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import scipy

inputDataFile = "SMSSpamCollection"
outputFile = 'results'

def tokenize(sms):
    """ Tokenizes to words as in Penn Treebank"""
    tokens = word_tokenize(sms.lower())
    return tokens

def ngrams(text, n):
    """ Returns character n-grams """
    return [text[i:i+n] for i in range(len(text)-n+1)]

def ParseFile(fileName):
    """ Parses the dataset file and returns the SMS and class lists."""
    smsList = list()
    categoryList = list()
    with open(inputDataFile, mode="r", encoding="utf-8") as file:
        for line in file:
            cat, sms = line.split('\t', maxsplit=1)
            smsList.append(sms)
            categoryList.append(cat)
            tokens = tokenize(sms)
    return smsList, categoryList

def VectorizeTfIdf(trainIndices, smsList, categoryList):
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=True, sublinear_tf=True, tokenizer=tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform(smsList)

    yVecList = list()

    for cat in categoryList:
        if cat == "ham":
            yVecList.append(0)
        elif cat == "spam":
            yVecList.append(1)

    xArray = sklearn_representation #.toarray()
    yArray = np.array(yVecList)

    return xArray, yArray

def VectorizeTfIdfCorrect(trainIndices, smsList, categoryList):
    smsTrain = [smsList[i] for i in trainIndices]
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=True, sublinear_tf=True, tokenizer=tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform(smsTrain)

    N = len(smsList)
    M = sklearn_representation.shape[1]
    xArray = scipy.sparse.csr_matrix((N, M))

    xVecList = list()
    yVecList = list()

    for cat in categoryList:
        if cat == "ham":
            yVecList.append(0)
        elif cat == "spam":
            yVecList.append(1)
    
    xArray[trainIndices,:] = sklearn_representation

    t = 0
    for i, sms in enumerate(smsList):
        if i not in trainIndices:
            newSmsList = list()
            newSmsList.append(sms)
            vecSparse = sklearn_tfidf.transform(newSmsList)
            xArray[i,:] = vecSparse

    yArray = np.array(yVecList)

    return xArray, yArray

def VectorizeTfIdfLong(trainIndices, smsList, categoryList):

    smsTrain = [smsList[i] for i in trainIndices]
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=True, sublinear_tf=True, tokenizer=tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform(smsTrain)

    xVecList = list()
    yVecList = list()

    for cat in categoryList:
        if cat == "ham":
            yVecList.append(0)
        elif cat == "spam":
            yVecList.append(1)
    
    t = 0
    for i, sms in enumerate(smsList):
        if i in trainIndices:
            vecSparse = sklearn_representation[t]
            vec = (vecSparse.toarray())[0]
            xVecList.append(vec)
            t = t + 1
        else:
            smsTrainNew = smsTrain[:]
            smsTrainNew.append(sms)
            
            sklearn_tfidf_new = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=True, sublinear_tf=True, tokenizer=tokenize)
            sklearn_representation_new = sklearn_tfidf_new.fit_transform(smsTrainNew)
            newVecSparse = sklearn_representation_new[len(smsTrain)]
            newVec = (newVecSparse.toarray())[0]
            xVecList.append(newVec)

    xArray = np.array(xVecList)
    yArray = np.array(yVecList)

    return xArray, yArray

def VectorizeWord2Vec(trainIndices, smsList, categoryList):
    """ Vectorizes SMS messages using Word2Vec. Only the training SMSs are used in training of Word2Vec model. """
    vecSize = size

    smsTokenizedList = list()

    for sms in smsList:
        smsTokenizedList.append(tokenize(sms))

    smsTrain = [smsTokenizedList[i] for i in trainIndices]

    model = Word2Vec(sentences=smsTrain, sg=sg, size=vecSize, window = window, min_count=1, seed = 1)

    xVecList = list()
    yVecList = list()

    for sms in smsTokenizedList:
        vec = np.zeros(vecSize)
        count = 0
        for token in sms:
            if(token in model):
                count = count + 1
                vec = vec + model[token]
        #if count != 0:
        #    vec = vec / count
        xVecList.append(vec)

    for cat in categoryList:
        if cat == "ham":
            yVecList.append(0)
        elif cat == "spam":
            yVecList.append(1)

    xArray = np.array(xVecList)
    yArray = np.array(yVecList)

    return xArray, yArray

def ClassifyAndEvaluate(classifier, xTrain, xTest, yTrain, yTest):
    """ The classification is done and the evaluation results are returned """
    classifier.fit(xTrain, yTrain)
    success = 0
    fail = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    num = 0
    for i, target in enumerate(yTest):
        a = classifier.predict(xTest[i].reshape((1,-1)))
        num = num + 1
        if target == 1:
            if a == 1:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if a == 1:
                fp = fp + 1
            else:
                tn = tn + 1
    tp = tp / num
    tn = tn / num
    fp = fp / num
    fn = fn / num
    return tp, tn, fp, fn

def accuracy(tp, tn, fp, fn):
    """ Function to calculate accuracy score"""
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc

def precision(tp, tn, fp, fn):
    """ Function to calculate precision score"""
    p = tp / (tp + fp)
    return p

def recall(tp, tn, fp, fn):
    """ Function to calculate recall score"""
    r = tp / (tp + fn)
    return r

def f1score(tp, tn, fp, fn):
    """ Function to calculate F1 score"""
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r)
    return f1

def KFoldClassifyAndEvaluate(classifier, smsList, categoryList, K):
    """ Evaluates the method and parameters with k-fold cross validation """
    testScores = list()
    tps = list()
    tns = list()
    fps = list()
    fns = list() 
    kf = KFold(n_splits=K, random_state=1, shuffle=True) 
    kf.get_n_splits(smsList)

    for split in kf.split(smsList):
        trainIndices = split[0]
        testIndices = split[1]    

        xArray, yArray = VectorizeWord2Vec(trainIndices, smsList, categoryList)     

        xTrain = xArray[trainIndices]
        xTest = xArray[testIndices]
        yTrain = yArray[trainIndices]
        yTest = yArray[testIndices]
        tp, tn, fp, fn = ClassifyAndEvaluate(classifier, xTrain, xTest, yTrain, yTest)
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        score = accuracy(tp, tn, fp, fn)
        testScores.append(score)
    
    averageTP = sum(tps)/len(tps)
    averageTN = sum(tns)/len(tns)
    averageFP = sum(fps)/len(fps)
    averageFN = sum(fns)/len(fns)
    return averageTP, averageTN, averageFP, averageFN

# Starting point of the program:
smsList, categoryList = ParseFile(inputDataFile)

for sg in [1]:
    for size in [64]:
        for window in [5]:
            for hs in [0]:
                for cbow_mean in [1]:

                    f = open(outputFile, 'a')
                    print("Starting with sg:{}, size:{}, window:{}, hs:{}, cbow_mean:{} \n".format(sg, size, window, hs, cbow_mean))
                    f.write("Starting with sg:{}, size:{}, window:{}, hs:{}, cbow_mean:{} \n".format(sg, size, window, hs, cbow_mean))
                    f.close()

                    starttime = time.time() 

                    clf = svm.SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                      max_iter=-1, probability=True, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
                    averageTP, averageTN, averageFP, averageFN = KFoldClassifyAndEvaluate(clf, smsList, categoryList, 10)
                    gecen_sure = time.time() - starttime

                    acc = accuracy(averageTP, averageTN, averageFP, averageFN)
                    r = recall(averageTP, averageTN, averageFP, averageFN)
                    p = precision(averageTP, averageTN, averageFP, averageFN)
                    f1 = f1score(averageTP, averageTN, averageFP, averageFN)

                    f = open(outputFile, 'a')
                    print("SVM TP: {}".format(averageTP))
                    f.write("SVM TP: {} \n".format(averageTP))
                    print("SVM TN: {}".format(averageTN))
                    f.write("SVM TN: {} \n".format(averageTN))
                    print("SVM FP: {}".format(averageFP))
                    f.write("SVM FP: {} \n".format(averageFP))
                    print("SVM FN: {}".format(averageFN))
                    f.write("SVM FN {} \n".format(averageFN))
                    print("SVM accuracy: {}".format(acc))
                    f.write("SVM accuracy: {} \n".format(acc))
                    print("SVM recall: {}".format(r))
                    f.write("SVM recall: {} \n".format(r))
                    print("SVM precision: {}".format(p))
                    f.write("SVM precision: {} \n".format(p))
                    print("SVM f1 score: {}".format(f1))
                    f.write("SVM f1 score: {} \n".format(f1))
                    print("Time elapsed: {}".format(gecen_sure))
                    f.write("Time elapsed: {} \n".format(gecen_sure))
                    f.close()




