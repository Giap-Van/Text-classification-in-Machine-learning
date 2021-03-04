from sklearn import preprocessing
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble

X_data = pickle.load(open('DL_dataset/X_data.pkl', 'rb'))
y_data = pickle.load(open('DL_dataset/Y_data.pkl', 'rb'))

X_test = pickle.load(open('DL_dataset/X_test.pkl', 'rb'))
y_test = pickle.load(open('DL_dataset/Y_test.pkl', 'rb'))


encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)

# lb = preprocessing.LabelBinarizer()
# y_data_n = lb.fit_transform(y_data)
# y_test_n = lb.fit_transform(y_test)

encoder.classes_

# basic IF-IDF
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data)
X_data_tfidf =  tfidf_vect.transform(X_data)
X_test_tfidf =  tfidf_vect.transform(X_test)

# N-gram level IF-IDF
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
tfidf_vect_ngram.fit(X_data)
X_data_tfidf_ngram =  tfidf_vect_ngram.transform(X_data)
X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

# Character level IF-IDF
tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 3))
tfidf_vect_ngram_char.fit(X_data)
X_data_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_data)
X_test_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_test)

# thuật toán SVD (singular value decomposition) nhằm mục đích giảm chiều dữ liệu 
# của ma trận mà chúng ta thu được, mà vẫn giữ nguyên được các thuộc tính của ma trận gốc ban đầu
svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_data_tfidf)

X_data_tfidf_svd = svd.transform(X_data_tfidf)
X_test_tfidf_svd = svd.transform(X_test_tfidf)

# N-gram SVD
svd_ngram = TruncatedSVD(n_components=300, random_state=42)
svd_ngram.fit(X_data_tfidf_ngram)

X_data_tfidf_ngram_svd = svd_ngram.transform(X_data_tfidf_ngram)
X_test_tfidf_ngram_svd = svd_ngram.transform(X_test_tfidf_ngram)

m = X_data_tfidf_ngram_svd.shape[0]
n = X_data_tfidf_ngram_svd.shape[1]
for i in range(m):
    for j in range(n):
        X_data_tfidf_ngram_svd[i][j] = X_data_tfidf_ngram_svd[i][j] + 1
m = X_test_tfidf_ngram_svd.shape[0]
n = X_test_tfidf_ngram_svd.shape[1]
for i in range(m):
    for j in range(n):
        X_test_tfidf_ngram_svd[i][j] = X_test_tfidf_ngram_svd[i][j] + 1

# Character level SVD
svd_ngram_char = TruncatedSVD(n_components=300, random_state=42)
svd_ngram_char.fit(X_data_tfidf_ngram_char)

X_data_tfidf_ngram_char_svd = svd_ngram_char.transform(X_data_tfidf_ngram_char)
X_test_tfidf_ngram_char_svd = svd_ngram_char.transform(X_test_tfidf_ngram_char)

m = X_data_tfidf_ngram_char_svd.shape[0]
n = X_data_tfidf_ngram_char_svd.shape[1]
for i in range(m):
    for j in range(n):
        X_data_tfidf_ngram_char_svd[i][j] = X_data_tfidf_ngram_char_svd[i][j] + 1
m = X_test_tfidf_ngram_char_svd.shape[0]
n = X_test_tfidf_ngram_char_svd.shape[1]
for i in range(m):
    for j in range(n):
        X_test_tfidf_ngram_char_svd[i][j] = X_test_tfidf_ngram_char_svd[i][j] + 1

def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    
    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
        
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)
    
        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))


# Naive Bayes
print("Naive Bayes với tfidf")
train_model(naive_bayes.MultinomialNB(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n, is_neuralnet=False)
########### kết quả Naive Bayes với tfidf:
# Validation accuracy:  0.7211404728789986
# Test accuracy:  0.7024677045379265

print("BernoulliNB với tfidf")
train_model(naive_bayes.BernoulliNB(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n, is_neuralnet=False)
########### kết quả BernoulliNB với tfidf
# Validation accuracy:  0.760778859527121
# Test accuracy:  0.7232527326929447

print("BernoulliNB với tfidf SVD")
train_model(naive_bayes.BernoulliNB(), X_data_tfidf_svd, y_data_n, X_test_tfidf_svd, y_test_n, is_neuralnet=False)
########### kết quả BernoulliNB với tfidf SVD
# Validation accuracy:  0.8393602225312935
# Test accuracy:  0.828668433256045


# Linear Classifier
print("Linear classifier với tfidf")
train_model(linear_model.LogisticRegression(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n, is_neuralnet=False)
########### kết quả Linear classifier với tfidf:
# Validation accuracy:  0.9082058414464534
# Test accuracy:  0.9016230539913879

print("Linear Classifier với tfidf_N-gram_level SVD")
train_model(linear_model.LogisticRegression(), X_data_tfidf_ngram_svd, y_data_n, X_test_tfidf_ngram_svd, y_test_n, is_neuralnet=False)
# kết quả Linear Classifier với tfidf_N-gram_level SVD:
# Validation accuracy:  0.8497913769123783
# Test accuracy:  0.7960417356740642

print("Linear Classifier với tfidf_Character_level SVD")
train_model(linear_model.LogisticRegression(), X_data_tfidf_ngram_char_svd, y_data_n, X_test_tfidf_ngram_char_svd, y_test_n, is_neuralnet=False)
# kết quả Linear Classifier với tfidf_Character_level SVD
# Validation accuracy:  0.885952712100139
# Test accuracy:  0.8608810864524677

# Support Vector Machine (SVM)
print("SVM với tfidf")
train_model(svm.SVC(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n, is_neuralnet=False)
# ################## kết quả SVM với tfidf:
# Validation accuracy:  0.13630041724617525
# Test accuracy:  0.121232196091421

print("SVM với tfidf_N-gram_level SVD")
train_model(svm.SVC(),  X_data_tfidf_ngram_svd, y_data_n, X_test_tfidf_ngram_svd, y_test_n, is_neuralnet=False)
# ########### kết quả SVM với tfidf_N-gram_level SVD
# Validation accuracy:  0.13630041724617525
# Test accuracy:  0.121232196091421

print("SVM với tfidf_Character_level SVD")
train_model(svm.SVC(),  X_data_tfidf_ngram_char_svd, y_data_n, X_test_tfidf_ngram_char_svd, y_test_n, is_neuralnet=False)
########### kết quả SVM với tfidf_Character_level SVD
# Validation accuracy:  0.13630041724617525
# Test accuracy:  0.12172904935409076

# Random Forest Classifier
print("Random Forest Classifier với tfidf svd")
train_model(ensemble.RandomForestClassifier(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n, is_neuralnet=False)
# kết quả Random Forest Classifier với tfidf svd
# Validation accuracy:  0.7635605006954103
# Test accuracy:  0.7186154355746937
