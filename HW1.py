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
import warnings

warnings.filterwarnings("ignore")

# Constants
WIDTH = 227
HEIGHT = 227
nSamples = 1087 

if os.name == "posix": #for MAC
        slash = "/"
        path = "./"
else:
        slash = "\\"
        path = ".\\"

        
sampleSize = 3*WIDTH*HEIGHT
labels = ["dog", "guitar", "house","person"]
label_color = {'dog': 'red', 'guitar': 'green', 'house': 'blue', 'person': 'magenta'}
scaler = prepro.StandardScaler()

'''
        Functions
'''

def importImages(flag):
        # Creazione della matrice di 1087 campioni per tutte le immagini dei cani
        i = 0
        j = 0
        y = []
        tmp = []
        if flag == 'all':
                for dir in os.listdir(path):
                        localPath = path+ slash + dir
                        if os.path.isdir(localPath):
                                for file in os.listdir(localPath):
                                        if file.endswith(".jpg"):
                                                img_data = np.asarray(Image.open(localPath + slash + file))
                                                tmp.append(img_data.ravel())
                                                y.append(j)
                                                i = i + 1
                                j = j + 1
        else:
                localPath = path + slash + flag
                if os.path.isdir(localPath):
                        for file in os.listdir(localPath):
                                if file.endswith(".jpg"):
                                        img_data = np.asarray(Image.open(localPath + slash + file))
                                        tmp.append(img_data.ravel())
                                        y.append(j)
                                        i = i + 1
                        j = j + 1
        
        return np.array(tmp), y

def showColorClasses(input_matrix):
        cvec = []
        #fig = plt.figure(figsize = (6,6))
        plt.subplots_adjust(hspace = 0.4, wspace = 1)

        scaled = scaler.fit_transform(input_matrix)
        x_t = PCA(2).fit_transform(scaled)
        for k in y:
                #z = labels[k]
                cvec.append(label_color[labels[k]])
        cvec = [label_color[labels[k]] for k in y]
        #plt.subplot(1,3,1)
        plt.scatter(x_t[:,0],x_t[:,1],c=cvec, s=4)
        plt.title("Scatter plot using PC(1,2)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()
        
        pca_tot = PCA()
        pca_tot = pca_tot.fit(x)
        components3_4 = pca_tot.components_[3:5]
        components10_11 = pca_tot.components_[10:12]

        pca_tot.components_ = components3_4
        x_3_4 = pca_tot.transform(x)
        #plt.subplot(1,3,2)
        plt.scatter(x_3_4[:,0],x_3_4[:,1],c=cvec, s=4)
        plt.title("Scatter plot using PC(3,4)")
        plt.xlabel("PC3")
        plt.ylabel("PC4")
        plt.show()
        
        pca_tot.components_ = components10_11
        x_10_11 = pca_tot.transform(x)
        #plt.subplot(1,3,3)
        plt.scatter(x_10_11[:,0],x_10_11[:,1],c=cvec, s=4)
        plt.title("Scatter plot using PC(10,11)")
        plt.xlabel("PC10")
        plt.ylabel("PC11")
        plt.show()
        
        #plt.show()
        #plt.clf()
        
def showImageWithPca(n_pca, input_matrix, nImg, show):
        pca_x = PCA(n_pca)
        scaled = scaler.fit_transform(input_matrix)
        projected = pca_x.fit_transform(scaled)
        x_inv = pca_x.inverse_transform(projected)
        x_inv = scaler.inverse_transform(x_inv)

        #print variance ratio
        val = np.sum(pca_x.explained_variance_ratio_)
        print("Variance converage for the first " + str(n_pca) + ": " + str(val)) #variance covered with this pca

        if (show):
                fig = plt.figure()
                fig.add_subplot(1,2,1)
                plt.imshow(np.reshape(input_matrix[nImg,:]/255.0,(227,227,3)))
                fig.add_subplot(1,2,2)
                plt.imshow(np.reshape(x_inv[nImg,:]/255.0,(227,227,3)))
                plt.show()
                #plt.clf()
        else:
                return x_inv

def showImageWithPcaLast(n_pca, input_matrix, nImg, show):
        #PCA last n computation
        pca_last_n = PCA()
        scaled = scaler.fit_transform(input_matrix)
        pca_last_n = pca_last_n.fit(scaled)
        components = pca_last_n.components_[-n_pca:]
        pca_last_n.components_ = components
        x_last_n = pca_last_n.transform(scaled)
        x_inv_last_n = pca_last_n.inverse_transform(x_last_n)
        x_inv_last_n = scaler.inverse_transform(x_inv_last_n)

        #print variance ratio
        val = np.sum(pca_last_n.explained_variance_ratio_)
        print("Variance converage for the last " + str(n_pca) + ": " + str(val)) #variance covered with this pca

        if(show):
                fig_last_n = plt.figure()
                fig_last_n.add_subplot(1,2,1)
                plt.imshow(np.reshape(input_matrix[nImg,:]/255.0,(227,227,3)))
                fig_last_n.add_subplot(1,2,2)
                plt.imshow(np.reshape(x_inv_last_n[nImg,:]/255.0, (227,227,3)))
                plt.show()
                #plt.clf()
        else:
                return x_inv_last_n

def showVarCoverageRateoWithPCA(n_pca, input_matrix):
        #number of components over variance coverage
        if n_pca>=0:
                pca_x = PCA(n_pca).fit(input_matrix)
        else:
                pca_x = PCA().fit(input_matrix)
        plt.plot(np.cumsum(pca_x.explained_variance_ratio_)) 
        plt.title("Variance retained in function of number of PCs")
        plt.xlabel('Number of Components') 
        plt.ylabel('Variance retained') 
        plt.ylim(0,1) 
        plt.grid()
        plt.show()
        #plt.clf()

def naiveBayesClassifier(input_matrix, classes, firstPC=0, lastPC=0):
        scaled = scaler.fit_transform(input_matrix)
        if firstPC == 0 and lastPC == 0:
                x_train, x_test, y_train, y_test = train_test_split(scaled, classes, test_size=0.1)
        else:
                #Select only PCA in range
                pca_tot = PCA()
                pca_tot = pca_tot.fit(scaled)
                twoComponents = pca_tot.components_[firstPC:lastPC]
                pca_tot.components_ = twoComponents
                x = pca_tot.transform(scaled)   
                #Train and use the model
                x_train, x_test, y_train, y_test = train_test_split(x, classes, test_size=0.1)
                
        clf = GaussianNB()
        clf.fit(x_train,y_train)
        prediction = clf.predict(x_test)
        accuracy = accuracy_score(y_test,prediction)
        print('Accuracy score: ' + str(accuracy))

def plotBoundaries(input_matrix, classes):
    cvec = []
    step = 20
    projected = PCA(2).fit_transform(input_matrix)
    
    x = projected[:, [0,1]]
    
    x_train, x_test, y_train, y_test = train_test_split(projected, classes, test_size=0.1)
    cvec = [label_color[labels[k]] for k in y_test]
    
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    #Plotting decision regions
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=cvec, s=10, edgecolor='k')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Init - import dataset
x, y = importImages('all')

#PCA transformations {1.2}
showVarCoverageRateoWithPCA(-1,x)

#Scatter plot
showColorClasses(x)

#Image reconstruction
image_list = []
title_list = []
image_list.append(x)
title_list.append("Original image")
image_list.append(showImageWithPca(60, x, 10, False))
title_list.append("First 60 PCs")

image_list.append(showImageWithPca(6, x, 10, False))
title_list.append("First 6 PCs")

image_list.append(showImageWithPca(2, x, 10, False))
title_list.append("First 2 PCs")

image_list.append(showImageWithPcaLast(6, x, 10, False))
title_list.append("Last 6 PCs")

figure = plt.figure()
plt.subplots_adjust(hspace = 0.2, wspace = 0.4)

for i in range(len(image_list)):
        figure.add_subplot(2, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(title_list[i])
        plt.imshow(np.reshape(image_list[i][10,:]/255.0,(227,227,3)))

plt.show()

#Naive bayes classifier accuracy scores
naiveBayesClassifier(x, y, firstPC=0, lastPC=0)
naiveBayesClassifier(x, y, firstPC=0, lastPC=2)
naiveBayesClassifier(x, y, firstPC=2, lastPC=4)

#Optional: boundaries plot
plotBoundaries(x, y)