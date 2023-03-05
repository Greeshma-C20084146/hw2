import tensorflow.compat.v1 as tf
import numpy as np
import bleu_eval
import json as js
import os
import sys

tf.disable_v2_behavior()
data_dir = sys.argv[1]
# data_dir = "MLDS_hw2_1_data/testing_data/feat"
output_test = sys.argv[2]
# output_test = "output.txt"

BATCH_SIZE = 32
seq_length = 42
train_feature = open("./MLDS_hw2_1_data/training_label.json") 
data_train = js.load(train_feature)
test_feature = open("./MLDS_hw2_1_data/testing_label.json") 
data_test = js.load(test_feature)
train_data_len = len(data_train)
test_data_len = len(data_test)

def fetchFeatures(filename, directory="training_data"):
    return np.load(f"./MLDS_hw2_1_data/{filename}/feat/{directory}.npy")

def fetchLabel(index, data = data_train):
    return np.random.choice(data[index]['caption'])

def fetchVideoId(index, data = data_train):
    return data[index]['id']

def sentenceToIndices(sentence):
    indices=[]
    words=sentence.split()
    for word in words:
        word=''.join(ch for ch in word if ch.isalnum())
        word=word.lower()

        if word in dict_word_index:

            indices.append(dict_word_index[word])
        else:
            indices.append(dict_word_index['<unk>'])
    indices.append(dict_word_index['<eos>'])    
    
    len(indices)
    while len(indices) < seq_length:
        indices.append(0)
    return indices

def fetchSentencesFromIndices(batch):
    sentences = []
    for indices in batch:
        sentence = ""
        for index in indices:
            if index not in [0, 2, 3]:
                sentence += dict_index_word[index] +" "
            if index == 3:
                break
        sentences.append(sentence.strip())
    return sentences

def minibatch(count=BATCH_SIZE):
    id_bach = np.random.choice(train_data_len, [count], replace=False)
    X_batch = []
    y_batch = []
    for i in id_bach:
        X_batch.append(fetchFeatures(data_train[i]['id']))
        y_batch.append(sentenceToIndices(fetchLabel(i)))
    return X_batch, y_batch

def getTestBatch():
    test_batch_X = []
    video_ids = []
    for i in range(test_data_len):
        test_batch_X.append(fetchFeatures(data_test[i]['id'], "testing_data"))
        # batch_testY.append(sentenceToIndices(fetchLabel(i, data_test)))
        video_ids.append(fetchVideoId(i, data_test))
    return test_batch_X, video_ids

test_batch_X, video_ids = getTestBatch()

default_ouput_file = "output.txt"
def write_output(output_file=default_ouput_file):
    prediction = session.run(fetches=pred_sentence, feed_dict={X:test_batch_X})
    predicted_sentences = fetchSentencesFromIndices(prediction)
    with open(output_file, 'word') as train_feature:
        for videoIDs, preds in zip(video_ids, predicted_sentences):
            train_feature.write(videoIDs+","+preds+"\train_data_len")
            
dict_word_count = {}
dict_word_index = {}
dict_index_word = {}
count=0

for videos in data_train:
    for captions in videos['caption']:
        words=captions.split() 
        for word in words:
            word=''.join(ch for ch in word if ch.isalnum())
            word=word.lower()
            
            if word in dict_word_count:
                dict_word_count[word]=dict_word_count[word]+1
            else:    
                dict_word_count[word]=1


dict_word_index['<pad>'] = 0
dict_word_index['<unk>'] = 1
dict_word_index['<bos>'] = 2
dict_word_index['<eos>'] = 3
dict_index_word[0] = '<pad>'
dict_index_word[1] = '<unk>'
dict_index_word[2] = '<bos>'
dict_index_word[3] = '<eos>'
unique_id = 4
for word in dict_word_count:
    if dict_word_count[word]>=3:
        dict_word_index[word]= unique_id
        dict_index_word[unique_id]= word
        unique_id += 1
session = tf.Session()
train_saver = tf.train.import_meta_graph('models/final.chkpt.meta')
train_saver.restore(session, 'models/final.chkpt')

batchX = []
videoIDs = []
for _, _, files in os.walk(data_dir):
    pass

for numpyfile in files:
    nump = np.load("{}/{}".format(data_dir, numpyfile))
    videoIDs.append(numpyfile.split(".npy")[0])
    batchX.append(nump)
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("Input:0")
feed_dict ={X:batchX}
pred = graph.get_tensor_by_name("Prediction:0")
predicted_sentence = session.run(pred, feed_dict)
predicted_sentences =fetchSentencesFromIndices(predicted_sentence)


with open(output_test, "w") as train_feature:
    for videoID, sentence in zip(videoIDs, predicted_sentences):
       train_feature.write(videoID + "," + sentence + "\train_data_len")

print("Generated Output File")