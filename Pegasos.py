import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import scipy.sparse




# Q 1.1
def read_training_data():
    file_location = 'trainingandtestdata/training.1600000.processed.noemoticon.csv'
    # file_location = 'trainingandtestdata/testing_cleaning_training_data'
    train = pd.read_csv(file_location,
                        delimiter="\",\"",
                        engine='python',
                        header=None,
                        names=['Sentiment', 'UserID', 'Date', 'NoQuery', 'Username', 'Tweet'])

    if False:  # change to False for full data set
        train_head = train.head(5000)
        train_tail = train.tail(5000)
        train = pd.concat([train_head, train_tail], axis=0)

    train['Sentiment'].replace('\"', '', regex=True, inplace=True)
    train['Sentiment'] = pd.to_numeric(train['Sentiment'])
    train['Sentiment'] = train['Sentiment'] // 2 - 1
    train['Tweet'] = train['Tweet'].str.split(',').str[0]  # because of the comma separator issue
    return train[['Sentiment', 'Tweet']]

#Q 1.6
def read_test_data():
    file_location = 'trainingandtestdata/testdata.manual.2009.06.14.csv'
    # file_location = 'trainingandtestdata/testing_cleaning_training_data'
    test = pd.read_csv(file_location,
                        delimiter="\",\"",
                        engine='python',
                        header=None,
                        names=['Sentiment', 'UserID', 'Date', 'NoQuery', 'Username', 'Tweet'])

    if False:  # change to False for full data set
        train_head = test.head(5000)
        train_tail = test.tail(5000)
        test = pd.concat([train_head, train_tail], axis=0)

    test['Sentiment'].replace('\"', '', regex=True, inplace=True)
    test['Sentiment'] = pd.to_numeric(test['Sentiment'])

    test.loc[test.Sentiment == 2] = 4

    test['Sentiment'] = test['Sentiment'] // 2 - 1
    test['Tweet'] = test['Tweet'].str.split(',').str[0]  # because of the comma separator issue
    return test[['Sentiment', 'Tweet']]

# Q 1.2
def clean_data(data):
    stop_words = np.loadtxt('stopwords.txt', delimiter='\n', dtype=str)
    stop_regex = '\\b('
    for word in stop_words:
        stop_regex += word + '|'
    stop_regex = stop_regex[:-1] + ')\\b'

    data['Tweet'] = data['Tweet'].str.lower()

    replacements1 = {r'(www\.\S*)|((http|https):\/\S*)': 'URL', r'\s{2,}': ' '}
    replacements2 = {r'([!"#\$%&\'\(\)\*\+,-\./;<=>\\?\[\]\^_`\{\|\}~:])|(^\s)|(\s$)': ''}
    replacements3 = {r'@\S*': 'AT-USER'}
    replacements4 = {r'\s{2,}': ' '}
    replacements5 = {r'(^\s)|(\s$)': ''}
    replacements6 = {stop_regex: ' '}
    data['Tweet'].replace(replacements1, regex=True, inplace=True)
    data['Tweet'].replace(replacements2, regex=True, inplace=True)
    data['Tweet'].replace(replacements3, regex=True, inplace=True)
    data['Tweet'].replace(replacements6, regex=True, inplace=True)
    data['Tweet'].replace(replacements4, regex=True, inplace=True)
    data['Tweet'].replace(replacements5, regex=True, inplace=True)

    tweet_df = data['Tweet'].str.split(' ', expand=False)
    tweet_df = tweet_df.apply(np.unique)

    replacements7 = {r'[\'\[\],]': ''}
    tweet_df = tweet_df.astype(str)
    tweet_df.replace(replacements7, regex=True, inplace=True)
    data = pd.concat([data['Sentiment'], tweet_df], axis=1)

    return data



# Q 1.3
def createFeatures(trainData,testData):

    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(trainData['Tweet'])
    scipy.sparse.save_npz('sparse_matrix_train_bigram.npz', train_features)

    newVec = CountVectorizer(vocabulary=vectorizer.vocabulary_)
    test_features = newVec.fit_transform(testData['Tweet'])
    scipy.sparse.save_npz('sparse_matrix_test_bigram.npz', test_features)

    trainLabels = trainData['Sentiment'].as_matrix()
    trainLabels = np.int8(trainLabels)
    trainLabels = trainLabels.reshape(trainLabels.shape[0],1)
    np.save('trainLabels',trainLabels)

    testLabels = testData['Sentiment'].as_matrix()
    testLabels = np.int8(testLabels)
    testLabels = testLabels.reshape(testLabels.shape[0],1)
    np.save('testLabels',testLabels)



def l2norm(vector):
    return math.sqrt(np.dot(vector.T, vector))

def l2norm2(vector):
    return math.sqrt(np.dot(vector, vector.T))



# Q 1.4
def pegasos_sparse(mat,reg,iters,batchSize,Y,test,Ytest):
    '''

    :param mat: sparse matrix of train features
    :param reg: regualarization
    :param iters: number of iterations
    :param batchSize:
    :param Y: Sentimant Labels Train
    :param test: test features
    :param Ytest: test labels
    :return: erros by iteration, test and train full dataset
    '''
    p = mat.shape[1]# number of features
    n = mat.shape[0] # number of samples
    ntest = test.shape[0]
    print "Features: ", p, "Samples: ", n
    w = np.zeros((1,p))
    w_old = np.zeros((1, p))
    numTrainErrs = np.zeros(iters)
    numTestErrs = np.zeros(iters)

    for i in range(iters):
        if batchSize < n:
            idx = np.random.randint(n, size=batchSize)
            batch = mat[idx, :]
            Ybatch = Y[idx,:]
        else:
            batch = mat
            Ybatch = Y

        A_t = batch[np.where(batch.dot(w.T) * Ybatch < 1)[0], :] # gets rows in X that are < 1 (wrong)
        Y_t = Ybatch[np.where(batch.dot(w.T) * Ybatch < 1)[0], :] # gets rows in Y that are < 1 (wrong)

        learningRate = float(1) /((i+1)*reg)
        #sumAt = np.sum(A_t.toarray()*Y_t ,axis=0) #y_i * x of ones that are wrong and then summed
        sumAt1 = scipy.sparse.coo_matrix.sum(A_t.multiply(Y_t), axis=0).T  # y_i * x_i of ones that are wrong and then sum by cols
        sumAt1 = sumAt1.A1 #make numpy array


        gradient = reg*w - (learningRate/(batchSize)) * sumAt1
        w_new = w - (learningRate)*gradient
        norm = l2norm2(w_new)
        val = (float(1) / math.sqrt(reg))/ norm  ###projection
        w = min(1,val) * w_new
        numTrainErrs[i] = np.where(mat.dot(w.T) * Y < 0)[0].shape[0] / float(n)
        numTestErrs[i]  = np.where(test.dot(w.T) * Ytest < 0)[0].shape[0] / float(ntest)

        if i % 50 == 0:
            # print  np.amin(w), np.amax(w)
            print  l2norm2(w - w_old)
            w_old = w
            print 'Iteration: ', i," Train Error: ", numTrainErrs[i], " Test Error: ", numTestErrs[i], " Batch", batchSize


    return numTrainErrs, numTestErrs




# Q 1.5, with additions for 1.6
def adaGrad(mat,reg,iters,batchSize,Y,test,Ytest):
    '''
    Combines Adaagrad and pegasos into one function. Also calculates training and test error on full dataset at each iteration
    :param mat: sparse matrix of train features
    :param reg: lambda
    :param iters: iterations
    :param batchSize: batch Size
    :param Y: Sentiment train Labels
    :param test: sparse matrix of test features
    :param Ytest: Sentiment test Labels
    :return: numTrainErrs - Adagrad training Errors
             numTestErrs - AdaGrad Test Errors
             numTrainErrsPeg - Pegasos Train Errors
             numTestErrsPeg - Pegasos Test Errors
    '''

    p = mat.shape[1]# number of features
    n = mat.shape[0] # number of train samples
    ntest = test.shape[0] # number of test samples
    print "Features: ", p, "Samples: ", n
    print "Test Features: ", test.shape[1], " Samples: ", ntest
    w = np.zeros((1,p))
    w_peg = np.zeros((1, p))
    numTrainErrs = np.zeros(iters)
    numTestErrs = np.zeros(iters)
    numTrainErrsPeg = np.zeros(iters)
    numTestErrsPeg = np.zeros(iters)

    G = np.ones((1,p))
    # G = G +  10**-10
    w_old = np.zeros((1,p))
    for i in range(iters):
        if batchSize < n:
            idx = np.random.randint(n, size=batchSize)
            batch = mat[idx, :]
            Ybatch = Y[idx,:]
        else:
            batch = mat
            Ybatch = Y

        A_t = batch[np.where(batch.dot(w.T) * Ybatch < 1)[0], :] # gets rows in X that are < 1 (wrong)
        Y_t = Ybatch[np.where(batch.dot(w.T) * Ybatch < 1)[0], :] # gets rows in Y that are < 1 (wrong)

        learningRate = float(1) /((i+1)*reg)
        sumAt1 = scipy.sparse.coo_matrix.sum(A_t.multiply(Y_t), axis=0).T  # y_i * x_i of ones that are wrong and then sum by cols
        sumAt1 = sumAt1.A1 #make numpy array


        gradient = reg*w - (learningRate/(batchSize)) * sumAt1

	    ##### AdaGrad Section ####

        G_new = np.power(gradient,2)
        G = G + G_new
        w_new = w - (learningRate/np.sqrt(G))*gradient
        norm = l2norm2(np.sqrt(G)*w_new)
        val = (float(1) / math.sqrt(reg))/ norm  ###projection
        w = min(1,val) * w_new
        numTrainErrs[i] = np.where(mat.dot(w.T) * Y < 0)[0].shape[0] / float(n)
        numTestErrs[i]  = np.where(test.dot(w.T) * Ytest < 0)[0].shape[0] / float(ntest)


        ###### Pegasos Section #####
        w_new_peg = w_peg - (learningRate)*gradient
        norm = l2norm2(w_new_peg)
        val = (float(1) / math.sqrt(reg))/ norm  ###projection
        w_peg = min(1,val) * w_new_peg

        numTrainErrsPeg[i] = np.where(mat.dot(w_peg.T) * Y < 0)[0].shape[0] / float(n)
        numTestErrsPeg[i] = np.where(test.dot(w_peg.T) * Ytest < 0)[0].shape[0] / float(ntest)


        if i % 10000 == 0:
           # print  'L2 Norm between Pegasos and Adagrad Weights', l2norm2(w - w_peg)
            print 'Iteration: ', i, " Train Error: ", numTrainErrs[i], " Test Error: ", numTestErrs[i]
            print 'Iteration: ', i, " Peg Train Error: ", numTrainErrsPeg[i], " Peg Test Error: ", numTestErrsPeg[i]

    return numTrainErrs, numTestErrs, numTrainErrsPeg, numTestErrsPeg


# Runs 1.1 through 1.3
def clean_create():

    print "Reading in training data..."
    start_time = time.time()

    train = read_training_data()
    test  = read_test_data()

    print ("COMPLETED: Reading in training data;  RunTime = %s seconds\n" % (time.time() - start_time))
    print "Cleaning training data..."

    start_time = time.time()

    train = clean_data(train)
    test = clean_data(test)
             
             
    print ("COMPLETED: Cleaning training data;  RunTime = %s seconds\n" % (time.time() - start_time))



    print "Making Features.."
    start_time = time.time()

    createFeatures(train,test)


    print ("COMPLETED: Making Features;  RunTime = %s seconds\n" % (time.time() - start_time))



def make_plots(iters,AdaTrainErrs, AdaTestErrs, PegaTrainErrs, PegaTestErrs,reg):
    x = np.arange(iters)
    plt.plot(x, AdaTrainErrs)
    plt.plot(x, AdaTestErrs)
    plt.plot(x, PegaTrainErrs)
    plt.plot(x, PegaTestErrs)
    plt.ylabel('Error Rate')
    plt.xlabel('Iterations')
    plt.title('Gradient Descent')
    plt.legend(['AdaTrainErrs', 'AdaTestErrs', 'PegaTrainErrs', 'PegaTestErrs'], loc='upper right')
    plt.savefig('Final'+ str(reg) + '.png')
    plt.close()

# Runs 1.4  through 1.6
def runGrad():
    '''
    Loads in data and runs adagrad and pegasos
    '''

    Ytrain   = np.load('trainLabels.npy')
    train = scipy.sparse.load_npz('sparse_matrix_train.npz')
    Ytest   = np.load('testLabels.npy')
    test = scipy.sparse.load_npz('sparse_matrix_test.npz')


    train = preprocessing.scale(train,with_mean=False, copy=False)
    test = preprocessing.scale(test,with_mean=False, copy=False)

    iters = 7500000
    batchSize = 30
    reg = .01
    AdaTrainErrs, AdaTestErrs, PegaTrainErrs, PegaTestErrs  = adaGrad(train,reg, iters, batchSize, Ytrain,test,Ytest)


    make_plots(iters,AdaTrainErrs, AdaTestErrs,PegaTrainErrs, PegaTestErrs,reg)

    print "AdaGrad Test Accuracy: ", 1 - AdaTestErrs[-1], " PEGASOS Test Accuracy: ", 1 - PegaTestErrs[-1]





if __name__ == "__main__":

    start_time = time.time()

    # Question 1.1 through 1.3
    # clean_create()

    # Run Pegasos and AdaGrad (1.4 - 1.6)
    runGrad()

    print ("COMPLETED: Pegasos;  RunTime = %s seconds\n" % (time.time() - start_time))
