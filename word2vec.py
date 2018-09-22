from nltk.corpus import gutenberg
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import string
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


emma = gutenberg.sents('austen-emma.txt')
corpus = [" ".join(sent) for sent in emma][100:150]
porter = PorterStemmer()

def preprocess(sents):
    for key, value in enumerate(sents):
        words = word_tokenize(value)
        words = [word.lower() for word in words]
        table = str.maketrans('', '', string.punctuation)
        stripped = [word.translate(table) for word in words]
        stop_words = set(stopwords.words('english'))
        stripped = [word for word in stripped if not word in stop_words]
        stripped = [porter.stem(word) for word in stripped]
        sents[key] = " ".join(stripped).strip(" ")
    return sents

corpus = preprocess(corpus)
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = set(words)

word2int = {}

for key,value in enumerate(words):
    word2int[value] = key

sentences = []
for sentence in corpus:
    sentences.append(sentence.split())

WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for key,value in enumerate(sentence):
        for neighbor in sentence[max(key - WINDOW_SIZE, 0) : min(key + WINDOW_SIZE, len(sentence)) + 1] : 
            if neighbor != value:
                data.append([value, neighbor])

df = pd.DataFrame(data, columns = ['input', 'label'])
print(df.head(10))

ONE_HOT_DIM = len(words)

def encode(ind):
    encoding = np.zeros(ONE_HOT_DIM)
    encoding[ind] = 1
    return encoding

X = []
Y = []

for x, y in zip(df['input'], df['label']):
    X.append(encode(word2int[x]))
    Y.append(encode(word2int[y]))

X_train = np.asarray(X)
Y_train = np.asarray(Y)

x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

EMBEDDING_DIM = 2

W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1]))
hidden_layer = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) 

iteration = 20000
for i in range(iteration):
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))

vectors = sess.run(W1 + b1)
print(vectors)

w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2']]


fig, ax = plt.subplots()

for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1,x2 ))
    
PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING
 
plt.xlim(x_axis_min,x_axis_max)
plt.ylim(y_axis_min,y_axis_max)
plt.rcParams["figure.figsize"] = (10,10)
plt.show()












