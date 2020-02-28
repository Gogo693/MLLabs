# pylint: disable=E1101
from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as prepro
from matplotlib import gridspec

from sklearn import datasets
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
import random

#load and split the data

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 1, stratify = y)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.6, random_state = 1, stratify = y_test)

svmVec = []
Cv = []
gammaV = []
c = 0.001
scoreLinVec = []
rbfVec = []
scoreRbfVec = []
rbfMatrix = np.empty([7, 13])
rbfScoreMatrix = np.empty([7, 13])
z = 0
rbfMaxScore = 0

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace = 0.6, wspace = 0.4)

gamm = 0.000000001
for i in range(13):
    gamm = round(gamm, 14-i)
    gammaV.append(gamm)
    gamm = gamm * 10

#Getting scores for both Linear and Rbf - every case
for i in range(7):
    clf = svm.SVC(kernel='linear', random_state = 1, C = c)
    rbf = svm.SVC(kernel='rbf', random_state=1, gamma = 'auto', C = c)
    Cv.append(c)
    c = c * 10

    gamm = 0.000000001
    for j in range(13):
        rbfG = svm.SVC(kernel='rbf', random_state=1, gamma=gamm, C = c)
        rbfG.fit(X_train, y_train)
        score = rbfG.score(X_val, y_val)
        
        rbfScoreMatrix[i][j] = score

        if score > rbfMaxScore:
            rbfMax = rbfG   
            rbfMaxC = i
            rbfMaxGamma = j
            rbfMaxScore = score

        gamm = gamm * 10

    clf.fit(X_train, y_train)
    rbf.fit(X_train, y_train)

    score = clf.score(X_val, y_val)
    scoreLinVec.append(score)

    score = rbf.score(X_val, y_val)
    scoreRbfVec.append(score)

    #adding to plot linear decision boundaries with different C
    plt.subplot(5, 3, i+1)
    plot_decision_regions(X_val, y_val, clf = clf, legend = 2)
    plt.title("Linear SVM C = " + str(Cv[i]))

    #adding to plot Rbf decision boundaries with different C and default gamma
    plt.subplot(5, 3, i+8)
    plot_decision_regions(X_val, y_val, clf = rbf, legend = 2)
    plt.title("Rbf kernel C = " + str(Cv[i]))

    svmVec.append(clf)
    rbfVec.append(rbf)

#show all plots for linear and rbf decision boundaries with different C and fixed gamma
plt.show()
#plt.clf()

param_range = np.logspace(-3, 3, 7)
plt.ylim(0.6, 0.9)
lw = 2

#Linear svm: plotting accuracy on the validation set with different C for
plt.semilogx(param_range, scoreLinVec, label="Training score", color="darkorange", lw=lw)
plt.grid()
plt.title("Linear svm accuracy")
plt.xlabel("C values")
plt.ylabel("Accuracy values")
plt.show()
#plt.clf()

max_score_lin = max(scoreLinVec)
max_index_lin = scoreLinVec.index(max_score_lin)

#Linear svm: plotting decision regions using the best C
svmVec[max_index_lin].score(X_test, y_test)
plot_decision_regions(X_test, y_test, clf = svmVec[max_index_lin])
plt.title("Best linear svm C=" + str(Cv[max_index_lin]) + " accuracy=" + str(round(max_score_lin, 2)))
plt.show()
#plt.clf()

max_score_rbf = max(scoreRbfVec)
max_index_rbf = scoreRbfVec.index(max_score_rbf)

#Rbf: plotting decision regions using the best C and default gamma 
rbfVec[max_index_rbf].score(X_test, y_test)
plot_decision_regions(X_test, y_test, clf = rbfVec[max_index_rbf])
plt.title("Best rbf C=" + str(Cv[max_index_rbf]) + " and gamma=default" + " accuracy=" + str(round(max_score_rbf, 2)))
plt.show()
#plt.clf()

#Rbf: plotting heatmap for grid search of best C and gamma values
plt.imshow(rbfScoreMatrix, interpolation = 'nearest', cmap = plt.cm.hot)
plt.title("Heatmap for Rbf grid search")
plt.xlabel('Gamma')
plt.ylabel('C')
plt.colorbar()
gammaV[4] = 0.00001
gammaV[5] = 0.0001
plt.xticks(np.arange(len(gammaV)), gammaV, rotation = 45)
plt.yticks(np.arange(len(Cv)), Cv)
plt.show()
print(gammaV)
#plt.clf()

#Rbf: plotting rbf decision regions for best C and gamma values
rbfMax.score(X_test, y_test)
plot_decision_regions(X_test, y_test, clf = rbfMax)
plt.title("Best rbf C=" + str(Cv[rbfMaxC]) + " and gamma=" + str(gammaV[rbfMaxGamma]) + " accuracy=" + str(round(rbfMaxScore, 2)))
plt.show()
#plt.clf()

'''K FOLD VALIDATION'''

X_train = np.concatenate((X_train, X_val),axis=0)
y_train = np.concatenate((y_train, y_val))

X_folds = np.array_split(X_train, 5)
y_folds = np.array_split(y_train, 5)


scores = np.empty((len(Cv), len(gammaV), 5))
rbfMaxScore = 0
c = 0.001

for i in range(7):

    gamm = 0.000000001
    for j in range(13):
        rbfG = svm.SVC(kernel='rbf', gamma=gamm, C = c)
        sumScore = 0

        for k in range(5):
        # We use 'list' to copy, in order to 'pop' later on
            Xk_train = list(X_folds)
            Xk_test = Xk_train.pop(k)
            Xk_train = np.concatenate(Xk_train)
            yk_train = list(y_folds)
            yk_test = yk_train.pop(k)
            yk_train = np.concatenate(yk_train)
            rbfG.fit(Xk_train, yk_train)
            sumScore += rbfG.score(Xk_test, yk_test)
        
        rbfScoreMatrix[i][j] = sumScore / 5

        if rbfScoreMatrix[i][j] > rbfMaxScore:
            rbfMax = rbfG
            rbfMaxC = i
            rbfMaxGamma = j
            rbfMaxScore = rbfScoreMatrix[i][j]

        gamm = gamm * 10
    c = c * 10


#K-fold: plotting decision regions for best C and gama values
rbfMax = svm.SVC(kernel='rbf', gamma=gammaV[rbfMaxGamma], C = Cv[rbfMaxC])
rbfMax.fit(X_train,y_train)
rbfMax.score(X_test, y_test)
plot_decision_regions(X_test, y_test, clf = rbfMax)
plt.title("Best k-fold C=" + str(Cv[rbfMaxC]) + " and gamma=" + str(gammaV[rbfMaxGamma]) + " accuracy=" + str(round(rbfMaxScore, 2)))
plt.show()
