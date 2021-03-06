import tensorflow as tf
import numpy as np
import csv
import random
from sklearn.decomposition import PCA

#Global Variable
n_input = 5
n_output = 5
n_hidden = [10,8]
n_epoch = 5000
learningRate = 0.76
modelResultValue = []
defaultResultValue = []
minFeature = []
maxFeature = []
save_dir = "./bpnn-model/"
filename = "bpnn.ckpt"

x = tf.placeholder(tf.float32, [None,n_input])
t = tf.placeholder(tf.float32, [None,n_output])

def loadRawData(filepath):
    result = []
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)
        for i in reader:
            result.append(i)
    return result

def getFeature(dataset):
    result = []
    for i in dataset:
        temp = i[5:11] + [i[12]] + i[14:16] + i[17:21] + [i[23]]
        result.append(temp)

    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            result[i][j] = float(result[i][j])

    return result

def getLabel(dataset):
    result = []
    for i in dataset:
        temp = i[4]
        result.append(temp)
    return result

def convertLabelToOneHotVector(datasetLabel):
    result = []
    for i in datasetLabel:
        temp = np.zeros(5, int)
        index = -1
        if(i == 'a'):
            index = 0
        elif(i == 'b'):
            index = 1
        elif(i == 'c'):
            index = 2
        elif(i == 'd'):
            index = 3
        elif(i == 'e'):
            index = 4
        temp[index] = 1

        result.append(temp)
    return result

def getMinMaxValue(dataset):
    maxFeature = [-1 for i in range(0, 14)]
    minFeature = [17000 for i in range(0, 14)]

    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[i])):
            if(dataset[i][j] > maxFeature[j]):
                maxFeature[j] = dataset[i][j]
            if(dataset[i][j] < minFeature[j]):
                minFeature[j] = dataset[i][j]
    
    return minFeature, maxFeature

def normalize(datasetFeature):
    result = datasetFeature.copy()
    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            result[i][j] = (result[i][j] - minFeature[j])/(maxFeature[j] - minFeature[j])

    return result    

def denormalize(datasetFeature):
    result = datasetFeature.copy()
    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            result[i][j] = result[i][j] * (maxFeature[j] - minFeature[j]) + minFeature[j]
    return result    

def applyPCA(datasetFeature):
    pca = PCA(n_components=5)
    dataset_pca = pca.fit_transform(datasetFeature)
    return dataset_pca 

def fullyConnected(input, n_input, n_output):
    w = tf.Variable(tf.random_normal([n_input, n_output]))
    b = tf.Variable(tf.random_normal([n_output]))
    linearCombination = tf.matmul(input,w) + b
    activationFunction = tf.nn.sigmoid(linearCombination)
    return activationFunction

def buildModel(input, n_input, n_hidden, n_output):
    x = fullyConnected(input,n_input,n_hidden[0])
    if(len(n_hidden) == 1):
        return fullyConnected(x, n_hidden[0], n_output)

    c = 0
    for i in range(0, len(n_hidden)-1):
        x = fullyConnected(x, n_hidden[i], n_hidden[i+1])
        c = i
    x = fullyConnected(x, n_hidden[c+1], n_output)
    return x

def optimize(model,trainDataset, validateDataset):
    
    loss = tf.reduce_mean(0.5 * (t-model)**2)
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
    modelResult = tf.argmax(model,1)
    defaultResult = tf.argmax(t,1)
    correct_prediction = tf.equal(modelResult, defaultResult)
    accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    valerrbefore = 100000000
    global modelResultValue
    global defaultResultValue
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())
        for epoch in range(n_epoch + 1):
            features = [i[0] for i in trainDataset]
            labels = [i[1] for i in trainDataset]

            feed_dict = {
                x:features,
                t:labels
            }
            _, lossVal, accuracyVal, modelResultValue, defaultResultValue = sess.run([optimizer, loss, accuracy, modelResult, defaultResult], feed_dict)
            if(epoch % 100 == 0):
                print(f"Epoch: {epoch} | Loss: {lossVal*100} | Accuracy: {accuracyVal*100} | modelResult: {modelResultValue} | defaultResult: {defaultResultValue}")
            
            if(epoch == 500):
                print(f"\n====================================================================================================")
                print(f"VALIDATION |Epoch: {epoch} | Loss: {validateErrVal*100} | Accuracy: {validateAccVal*100} | modelResult: {validateResultVal} | defaultResult: {defaultResultVal}")
                print(f"====================================================================================================\n")

            if(epoch % 500 == 0):
                validateErrVal, validateAccVal, validateResultVal, defaultResultVal= sess.run([loss, accuracy, modelResult, defaultResult], feed_dict={
                    x: [j[0] for j in validateDataset],
                    t: [j[1] for j in validateDataset]
                })
                
                if(validateErrVal < valerrbefore):
                    saver.save(sess, save_dir + filename)
                    valerrbefore = validateErrVal

        # same = 0
        # for i in range(0, len(modelResultValue)):
        #     if(modelResultValue[i] == defaultResultValue[i]):
        #         same = same + 1
        # print(same, (same/len(modelResultValue)) * 100)

def test(model, testDataset):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        checkPoint = tf.train.get_checkpoint_state(save_dir)
        saver.restore(sess, checkPoint.model_checkpoint_path)

        modelResult = tf.argmax(model,1)
        defaultResult = tf.argmax(t,1)
        correct_prediction = tf.equal(modelResult, defaultResult)
        accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        accuracyVal, modelResultValue, defaultResultValue = sess.run([accuracy, modelResult, defaultResult], feed_dict={
            x: [i[0] for i in testDataset],
            t: [i[1] for i in testDataset]
        })
        
        y = 0
        for i in range(0, len(testDataset)):
            if(modelResultValue[i] == defaultResultValue[i]):
                y = y + 1

        print(f"Test result: {resultToAlphabetic(modelResultValue)}\nDefault result: {resultToAlphabetic(defaultResultValue)}")
        print(f"Total test data: {len(testDataset)}")
        print(f"Number of correct prediction: {y}")
        print(f"Test accuracy: {round(accuracyVal*100,1)}%")


def resultToAlphabetic(resultArr):
    result = []
    for i in range(0, len(resultArr)):
        if(resultArr[i] == 0):
            result.append('a')
        elif(resultArr[i] == 1):
            result.append('b')
        elif(resultArr[i] == 2):
            result.append('c')
        elif(resultArr[i] == 3):
            result.append('d')
        elif(resultArr[i] == 4):
            result.append('e')
    
    return np.array(result)
            
def mergeFeatureLabel(datasetFeature, datasetLabel):
    result = []
    for i in range(0, len(datasetFeature)):
        result.append((datasetFeature[i], datasetLabel[i]))
    return result

def divideFeatureLabel(dataset):
    features = []
    labels = []
    for i in dataset:
        features.append(i[0])
        labels.append(i[1])
    return features, labels

def main():
    rawDataset = loadRawData("O192-COMP7117-AS01-00-classification.csv")
    perc70 = int(0.7 * len(rawDataset))
    perc20 = int(0.2 * len(rawDataset))
    
    datasetFeature = getFeature(rawDataset)
    datasetLabel = getLabel(rawDataset)
    datasetLabel = convertLabelToOneHotVector(datasetLabel)
    
    oldFixDataset = mergeFeatureLabel(datasetFeature, datasetLabel)
    random.shuffle(oldFixDataset)
    
    oldFixFeatures, oldFixLabels = divideFeatureLabel(oldFixDataset)
    global minFeature
    global maxFeature
    minFeature, maxFeature = getMinMaxValue(oldFixFeatures)
    datasetFeature = normalize(oldFixFeatures)

    #===================================================================

    datasetFeaturePCA = applyPCA(datasetFeature)
    datasetPCA = mergeFeatureLabel(datasetFeaturePCA, oldFixLabels)
    

    trainDatasetPCA = datasetPCA[0:perc70]
    validateDatasetPCA = datasetPCA[perc70:(perc20 + perc70)]
    testDatasetPCA = datasetPCA[(perc20 + perc70):]

    model = buildModel(x, n_input, n_hidden, n_output)
    optimize(model,trainDatasetPCA, testDatasetPCA)
    test(model, testDatasetPCA)


main()

