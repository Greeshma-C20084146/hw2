import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import json as js
import bleu_eval

tf.disable_v2_behavior()

BATCH_SIZE = 128
seq_length = 42
train_feature = open("./MLDS_hw2_1_data/training_label.json") 
data_train = js.load(train_feature)
test_feature = open("./MLDS_hw2_1_data/testing_label.json") 
data_test = js.load(test_feature)
train_data_len = len(data_train)
test_data_len = len(data_test)

def fetchFeatures(filename, directory="training_data"):
    return np.load(f"./MLDS_hw2_1_data/{directory}/feat/{filename}.npy")

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
    with open(output_file, 'w') as train_feature:
        for videoIDs, preds in zip(video_ids, predicted_sentences):
            train_feature.write(videoIDs+","+preds+"\n")

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

tf.compat.v1.reset_default_graph()
hidden_layer = 128
X = tf.compat.v1.placeholder(tf.float32, shape = [None, 80,4096], name = 'Input')
Y = tf.compat.v1.placeholder(tf.int32, shape = [None, seq_length], name = 'Output')
batch_size = tf.shape(X)[0]
padding = tf.zeros(shape=[batch_size, hidden_layer])

lstm1= tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_layer)
lstm2= tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_layer)
state1=lstm1.zero_state(batch_size,dtype=tf.float32)
state2=lstm2.zero_state(batch_size,dtype=tf.float32)

vocab_len= len(dict_word_index)
embedding_size =hidden_layer
word_embedding=tf.Variable(tf.random.truncated_normal([vocab_len,embedding_size]))

for i in range(80):
    input_lstm1_enc= X[:,i,:]
    output_1_enc, state1= lstm1(input_lstm1_enc,state1)
    
    input_lstm2_enc = tf.concat([output_1_enc,padding],1)
    output_2_enc, state2= lstm2(input_lstm2_enc,state2)

pad=tf.zeros([batch_size,4096])
bos = tf.fill([batch_size], 2)
word_predict_embedding = tf.nn.embedding_lookup(word_embedding, bos)
loss = tf.zeros(batch_size)
pred_sentence = tf.fill([1, batch_size], 2)
for i in range(seq_length):
    output_lstm1_dec, state1= lstm1(pad,state1)    
    input_lstm2_dec=tf.concat([output_lstm1_dec,word_predict_embedding],1)
    output_lstm2_dec,state2=lstm2(input_lstm2_dec,state2)
        
    y_at_time_i = Y[:, i]
    y_oneHotEnc = tf.one_hot(indices=y_at_time_i, depth=vocab_len)
    logits = tf.matmul(output_lstm2_dec, tf.transpose(word_embedding))
    predicted_word_index=tf.argmax(logits, axis=1, output_type=tf.int32)
    word_predict_embedding=tf.nn.embedding_lookup(word_embedding,predicted_word_index)
    pred_sentence = tf.concat([pred_sentence, tf.expand_dims(predicted_word_index, axis=0)], axis=0)
    loss_of_this_word = tf.nn.softmax_cross_entropy_with_logits(labels=y_oneHotEnc, logits=logits)
    loss += loss_of_this_word
pred_sentence = tf.transpose(pred_sentence, name="Prediction")
loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
session = tf.Session()
session.run(tf.global_variables_initializer())
prev_loss = 0
intervals = 5
log_intervals = 50
best_bleu_score = 0.5
train_saver = tf.train.Saver()
for i in range(500):
    X_batch, y_batch = minibatch()
    l, test= session.run(fetches=[loss, optimizer], feed_dict={X:X_batch, Y:y_batch})
    prev_loss += l
    if i % intervals == 0:
        write_output(default_ouput_file)
        current_bleu_score = bleu_eval.fetchBlueScore(default_ouput_file)
        if (current_bleu_score > best_bleu_score):
            best_bleu_score = current_bleu_score
            write_output("best_ouput.txt")
            train_saver.save(session, "models/final.chkpt")
            print("Updated Best Score: ", best_bleu_score)
            
    if i % log_intervals == 0:
        print(i, prev_loss/log_intervals, best_bleu_score)
        prev_loss = 0

prediction = session.run(fetches=pred_sentence, feed_dict={X:X_batch, Y:y_batch})

# np.shape(prediction)

X_batch, y_batch = minibatch()

# y_batch
# fetchSentencesFromIndices(y_batch)
train_saver = tf.train.Saver()
train_saver.save(session, "training/checkpoint.chkpt")
train_saver.save(session, "training2/checkpoint.chkpt")
bleu_eval.fetchBlueScore(default_ouput_file)
