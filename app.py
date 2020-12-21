# Automated Essay Scoring

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import re
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors

app = Flask(__name__)

###############################################################################
lstm_model = tf.keras.models.load_model("lstmmodel.h5")
word2vec_model = KeyedVectors.load_word2vec_format('word2vecmodel.bin',binary=True)

# Preprocessing of input essay
def preprocess(X):
    corpus = []
    essay = re.sub('[^a-zA-Z]',' ',X) # Removes every char except [^a-zA-Z] 
    essay = essay.lower() # All characters in lower
    essay = essay.split() # split all the words
    lemmatizer = WordNetLemmatizer() # Lemmatize every word
    essay = [lemmatizer.lemmatize(word) for word in essay if not word in set(stopwords.words('english'))]
    essay = ' '.join(essay) # Join all the words
    corpus.append(essay)
        
    sent = []
    for i in range(len(corpus)):
        sent.append(nltk.sent_tokenize(corpus[i])) # Change all para into list of sentences
        
    word = []
    for i in range(len(sent)):
        word.append(nltk.word_tokenize(sent[i][0])) # Change all the sent into list of words
        
    return word

def makeFeatureVec(words, model, num_features=150):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,))
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model[word])        
    featureVec = np.divide(featureVec,num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features=150):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features))
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

def reshape(X):
    X = np.reshape(X,(X.shape[0],1,X.shape[1]))
    return X
###############################################################################

###############################################################################
@app.route('/')
def home():
    return render_template('home.html')
###############################################################################
    
###############################################################################
@app.route('/set1',methods=['GET','POST'])
def set1():
    if request.method=='POST':
        mini = 2
        maxi = 12
        essay = str(request.form['essay'])
        word = preprocess(essay)
        X = getAvgFeatureVecs(word,word2vec_model)
        X = reshape(X)
        y = lstm_model.predict(X)
        z = y[0][0]
        y = np.round(y[0][0])
        if y<mini:
            y=mini
        if y>maxi:
            y=maxi
        return render_template('set1.html', result=f"You got {y} marks. (rounded off from {z})",
                               min=f"Minimum marks in this set is {mini}.",
                               max=f"Maximum marks in this set is {maxi}.")
    
    return render_template('set1.html')
###############################################################################
    
###############################################################################
@app.route('/set2',methods=['GET','POST'])  
def set2():
    if request.method=='POST':
        mini = 1
        maxi = 6
        essay = str(request.form['essay'])
        word = preprocess(essay)
        X = getAvgFeatureVecs(word,word2vec_model)
        X = reshape(X)
        y = lstm_model.predict(X)
        z = y[0][0]
        y = np.round(y[0][0])
        if y<mini:
            y=mini
        if y>maxi:
            y=maxi
        return render_template('set2.html', result=f"You got {y} marks. (rounded off from {z})",
                               min=f"Minimum marks in this set is {mini}.",
                               max=f"Maximum marks in this set is {maxi}.")
    
    return render_template('set2.html')
###############################################################################
    
###############################################################################
@app.route('/set3',methods=['GET','POST'])  
def set3():
    if request.method=='POST':
        mini = 0
        maxi = 3
        essay = str(request.form['essay'])
        word = preprocess(essay)
        X = getAvgFeatureVecs(word,word2vec_model)
        X = reshape(X)
        y = lstm_model.predict(X)
        z = y[0][0]
        y = np.round(y[0][0])
        if y<mini:
            y=mini
        if y>maxi:
            y=maxi
        return render_template('set3.html', result=f"You got {y} marks. (rounded off from {z})",
                               min=f"Minimum marks in this set is {mini}.",
                               max=f"Maximum marks in this set is {maxi}.")
    
    return render_template('set3.html')  
###############################################################################
    
###############################################################################
@app.route('/set4',methods=['GET','POST'])  
def set4():
    if request.method=='POST':
        mini = 0
        maxi = 3
        essay = str(request.form['essay'])
        word = preprocess(essay)
        X = getAvgFeatureVecs(word,word2vec_model)
        X = reshape(X)
        y = lstm_model.predict(X)
        z = y[0][0]
        y = np.round(y[0][0])
        if y<mini:
            y=mini
        if y>maxi:
            y=maxi
        return render_template('set4.html', result=f"You got {y} marks. (rounded off from {z})",
                               min=f"Minimum marks in this set is {mini}.",
                               max=f"Maximum marks in this set is {maxi}.")
    
    return render_template('set4.html')   
###############################################################################
    
###############################################################################
@app.route('/set5',methods=['GET','POST'])  
def set5():
    if request.method=='POST':
        mini = 0
        maxi = 4
        essay = str(request.form['essay'])
        word = preprocess(essay)
        X = getAvgFeatureVecs(word,word2vec_model)
        X = reshape(X)
        y = lstm_model.predict(X)
        z = y[0][0]
        y = np.round(y[0][0])
        if y<mini:
            y=mini
        if y>maxi:
            y=maxi
        return render_template('set5.html', result=f"You got {y} marks. (rounded off from {z})",
                               min=f"Minimum marks in this set is {mini}.",
                               max=f"Maximum marks in this set is {maxi}.")
    
    return render_template('set5.html')   
###############################################################################
    
###############################################################################
@app.route('/set6',methods=['GET','POST'])  
def set6():
    if request.method=='POST':
        mini = 0
        maxi = 4
        essay = str(request.form['essay'])
        word = preprocess(essay)
        X = getAvgFeatureVecs(word,word2vec_model)
        X = reshape(X)
        y = lstm_model.predict(X)
        z = y[0][0]
        y = np.round(y[0][0])
        if y<mini:
            y=mini
        if y>maxi:
            y=maxi
        return render_template('set6.html', result=f"You got {y} marks. (rounded off from {z})",
                               min=f"Minimum marks in this set is {mini}.",
                               max=f"Maximum marks in this set is {maxi}.")
    
    return render_template('set6.html')   
###############################################################################
    
###############################################################################
@app.route('/set7',methods=['GET','POST'])  
def set7():
    if request.method=='POST':
        mini = 0
        maxi = 30
        essay = str(request.form['essay'])
        word = preprocess(essay)
        X = getAvgFeatureVecs(word,word2vec_model)
        X = reshape(X)
        y = lstm_model.predict(X)
        z = y[0][0]
        y = np.round(y[0][0])
        if y<mini:
            y=mini
        if y>maxi:
            y=maxi
        return render_template('set7.html', result=f"You got {y} marks. (rounded off from {z})",
                               min=f"Minimum marks in this set is {mini}.",
                               max=f"Maximum marks in this set is {maxi}.")
    
    return render_template('set7.html')   
###############################################################################
    
###############################################################################
@app.route('/set8',methods=['GET','POST'])  
def set8():
    if request.method=='POST':
        mini = 0
        maxi = 60
        essay = str(request.form['essay'])
        word = preprocess(essay)
        X = getAvgFeatureVecs(word,word2vec_model)
        X = reshape(X)
        y = lstm_model.predict(X)
        z = y[0][0]
        y = np.round(y[0][0])
        if y<mini:
            y=mini
        if y>maxi:
            y=maxi
        return render_template('set8.html', result=f"You got {y} marks. (rounded off from {z})",
                               min=f"Minimum marks in this set is {mini}.",
                               max=f"Maximum marks in this set is {maxi}.")
    
    return render_template('set8.html')   
###############################################################################

if __name__ == '__main__':
    app.run(use_reloader=False)

