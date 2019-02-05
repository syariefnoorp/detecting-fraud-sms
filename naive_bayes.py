#Will be executed after Preprocessing

import numpy as np
import preprocessing as pre
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

v = abs(len(pre.getFeature))

def rawTF(trainingDocument, label):

    temp = []
    stopword = StopWordRemoverFactory().get_stop_words()
    countVectorize = CountVectorizer(stop_words=stopword)

    countVectorize.fit_transform(trainingDocument['Teks'])

    getLabel = trainingDocument[trainingDocument['label'] == label]['Teks']

    fitFeatureStemming = countVectorize.transform(getLabel)

    getFeature = countVectorize.get_feature_names()

    arrayLabel = np.transpose(fitFeatureStemming.toarray())

    for i in range(len(getFeature)):
        sum = 0
        for j in range(len(arrayLabel[label])):
            sum += arrayLabel[i].item(j)
        temp.append((getFeature[i], sum))
    return temp
# print('Raw TF', rawTF(pre.stemmingDocument, 1))

def countTotalTFperClass(data):
    sum = 0
    for i in range(len(data)):
        sum += data[i][1]
    return sum

def countTotalTermPerClass(data, smsTest):
    temp = []
    splitSMS = smsTest.split(" ")
    for i in range(len(data)):
        for j in range(len(splitSMS)):
            if splitSMS[j] == data[i][0]:
                temp.append((splitSMS[j], data[i][1]))
    return temp

def multinomial (a ,b , c):
    x = []
    for i in range(len(a)):
        x.append((a[i][0], (a[i][1] + 1)/(b + c)))
    return x

#Calculate Prior
def probability(label):
    totalSMS = len(pre.docTraining)
    prob = (len(pre.docTraining[pre.docTraining['label'] == label])) / totalSMS
    return prob

#get count(w, c) => word per class
def getCountWC(label, sms):
    return countTotalTermPerClass(rawTF(pre.stemmingDocument, label), sms)

def calculateNaiveBayes(label, sms):
    temp = 1
    for i in range(len(getCountWC(label, sms))):
        temp *= (multinomial(getCountWC(label, sms), countTotalTFperClass(rawTF(pre.stemmingDocument, label)), v)[i][1]) #Get the Posterior
    result = probability(label) * temp
    return result

sms = 'PALING HEMAT! Kuota 2.5GB (24jam) semua jaringan Rp2000. Balas SMS ini ketik LW utk beli. Berlaku perpanjangan. Terus ISI PULSA utk KEJUTAN lainnya! AA021'
sms = sms.lower()

labelClass = [0, 1, 2]

def determineClass(sms):
    temp = []
    for i in range(len(labelClass)):
        temp.append(calculateNaiveBayes(i, sms))

    print('Result each class', temp)

    for i in range(len(labelClass)):
        if(temp[i] == max(temp)):
            print(labelClass[i],'\n')
            return (labelClass[i])

print('Kelas' , determineClass(sms))

read = pre.readDocument('data_testing.csv')

def accuracy(data):
    counter = 0
    totalSMSTesting = len(data)
    for i in range(len(data)):
        sms = data.iloc[i]['Teks'].lower()
        classResult = determineClass(sms)
        if classResult == data.iloc[i]['label']:
            counter += 1

    accuracyPercentage = (counter/totalSMSTesting) * 100
    return accuracyPercentage
print(accuracy(read))
