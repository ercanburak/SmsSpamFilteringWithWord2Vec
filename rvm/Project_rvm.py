import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import time
from collections import Counter
import numpy as np
from skrvm import RVC
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

starttime = time.time() 

#nltk.download()
#nltk.download("words")
#nltk.download("stopwords")
#nltk.download("punkt")

inputDataFile = "SMSSpamCollection"

smsList = list()
smsListTokenized = list()
all_features = list()
smsListemiz = list()
vecList = list()
targetList = list()

#stoplist = stopwords.words("english")

def preprocess(sms):
    tokens = word_tokenize(sms.lower())
    return tokens

with open(inputDataFile, mode="r", encoding="utf-8") as file:
    for line in file:
        sms = line.split('\t', maxsplit=1)
        smsList.append(sms)

for sms in smsList:
    smsListemiz.append(preprocess(sms[1]))
    smsListTokenized.append((sms[0], preprocess(sms[1])))

tokenCounts = Counter()

for (cat, smsTokens) in smsListTokenized:
    for token in smsTokens:
        tokenCounts[token] += 1

# Construct the vocabulary hash map by giving each word a unique number:
vocabulary = {}
V = 0
for token in tokenCounts:
    vocabulary[token] = V
    V += 1

del tokenCounts

model = gensim.models.Word2Vec(smsListemiz, min_count=1)

vec = np.zeros(100)

for sms in smsListTokenized:
    vec = np.zeros(100)
    for token in sms[1]:
        vec = vec + model[token]
    vecList.append(vec)
    if sms[0] == "ham":
        targetList.append(0)
    elif sms[0] == "spam":
        targetList.append(1)
    else:        
        print("Error")


vecArray = np.array(vecList)
targetArray = np.array(targetList)


clf = RVC()

clf.fit(vecArray[0:3000,:], targetArray[0:3000])
RVC(alpha=1e-06, beta=1e-06, beta_fixed=False, bias_used=True, coef0=0.0,
coef1=None, degree=3, kernel='rbf', n_iter=3000, n_iter_posterior=50,
threshold_alpha=1000000000.0, tol=0.001, verbose=False)
print(clf.score(vecArray[0:3000,:], targetArray[0:3000]))
print("test finished")


gecen_sure = time.time() - starttime
print("Time elapsed: {}".format(gecen_sure))