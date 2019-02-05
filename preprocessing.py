import numpy as np
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

#Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#Filter
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

#Variable initialization
documentName = 'dataset_sms_ori.csv'
totalClass = np.zeros(3)

#Method for Document Parsing
def readDocument(documentName):
    df = pandas.read_csv(documentName)
    return df

def getRowOfEveryClass(totalClass, document):
    for i in range(len(totalClass)):
        totalClass[i] = len(document[document['label'] == i])
    return totalClass

def case_folding(document):
    document = document['Teks'].str.lower()
    return document

def cleansing(document):
    #number
    document = document.str.replace('\d','')

    #link/url
    document = document.str.replace('((http(s)?)[0-9a-z\./_+\(\)\$\#\&\!\?]+)','')
    document = document.str.replace('((www?)[0-9a-z\./_+\(\)\$\#\&\!\?]+)','')
    document = document.str.replace('((tsel?\.)[0-9a-z\./_+\(\)\$\#\&\!\?]+)','')

    #simbol
    document = document.str.replace('[\\.()?,!""'':;/+=*#%\[\]]','')
    document = document.str.replace('[-_&]',' ')

    #emot
    document = document.str.replace('["\U0001F600-\U0001F64F" | "\U0001F300-\U0001F5FF"]+',' ')

    #fix space
    document = document.str.replace('\s+', ' ')

    #remove loop alphabet
    document = document.str.replace(r'([a-z])\1+',r'\1',regex=True)

    return document

def set_label(docLabel):
    labelArray = []
    for i in range(len(docLabel)):
        labelArray.append(docLabel.iloc[i])
    return labelArray

def stemming(docTraining):
    stemmingArray = []
    for i in range(len(docTraining)):
        stemmingResult = docTraining.iloc[i]
        stemmingArray.append(stemmer.stem(stemmingResult))
    return stemmingArray

def preprocessing(document):
    prepArray = []
    document = case_folding(document)
    document = cleansing(document)
    prepArray = stemming(document)
    return prepArray
    
#Read Document
document = readDocument(documentName)

#Filter Based on Document Label
docClass0 = document[document['label'] == 0]
docClass1 = document[document['label'] == 1]
docClass2 = document[document['label'] == 2]

#Create document with 20% of each class for Data Testing
docTest0 = docClass0[-(int(getRowOfEveryClass(totalClass, document)[0] * 0.2) ) :  ]
docTest1 = docClass1[-(int(getRowOfEveryClass(totalClass, document)[1] * 0.2) ) :  ]
docTest2 = docClass2[-(int(getRowOfEveryClass(totalClass, document)[2] * 0.2) ) :  ]

#Create document with 80% of each class
docTraining0 = docClass0[ : (int(getRowOfEveryClass(totalClass, document)[0] * 0.8) ) ]
docTraining1 = docClass1[ : (int(getRowOfEveryClass(totalClass, document)[1] * 0.8) ) ]
docTraining2 = docClass2[ : (int(getRowOfEveryClass(totalClass, document)[2] * 0.8) ) ]

#Data Training Document
docTraining = pandas.concat([docTraining2, docTraining1, docTraining0])
docTraining.to_csv("data_training.csv")

print('\nTraining Document', docTraining)

#Data Testing Document
docTesting = pandas.concat([docTest2, docTest1, docTest0])
docTesting.to_csv("data_testing.csv")

#print('\nTesting Document', docTesting)

#Preprocessing Document Training
'''docTrainingLabel = docTraining['label']
labelArray = set_label(docTrainingLabel)

stemmingArray = preprocessing(docTraining)

docStemmingTraining = pandas.DataFrame(data=stemmingArray, columns=['Teks'])
labelDataFrame = pandas.DataFrame(data=labelArray, columns=['label'])

docStemmingTraining = pandas.concat([docStemmingTraining, labelDataFrame], axis=1)
docStemmingTraining.to_csv('training_stemming.csv')
'''
#Read Stemming Document
#stemmingDocument = readDocument('training_stemming.csv')


#Merge with the Label
#docTraining = pandas.concat([docTraining, stemmingDocument['label']], axis=1)
#print('\nDocument Testing\n', docTraining)

#docTraining.to_csv('preprocessing_document.csv')

#Preprocessing Data Testing
docTestingLabel = docTesting['label']
labelArray = set_label(docTestingLabel)

stemmingArray = preprocessing(docTesting)

docStemmingTesting = pandas.DataFrame(data=stemmingArray, columns=['Teks'])
labelDataFrame = pandas.DataFrame(data=labelArray, columns=['label'])

docStemmingTesting = pandas.concat([docStemmingTesting, labelDataFrame], axis=1)
docStemmingTesting.to_csv('testing_stemming.csv')

#Read Stemming Document
stemmingDocument = readDocument('testing_stemming.csv')

#Tokenizing and Get Term
stemmingResult = stemmingDocument['Teks']
stopword = StopWordRemoverFactory().get_stop_words()
countVectorize = CountVectorizer(stop_words=stopword)
fitFeature = countVectorize.fit_transform(stemmingResult)
getFeature = countVectorize.get_feature_names()
arrayFeature = fitFeature.toarray()
print(getFeature)
