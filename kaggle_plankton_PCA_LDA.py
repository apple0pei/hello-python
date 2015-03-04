# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 11:57:11 2015

@author: applepei
"""

#Import libraries for doing image analysis
#%%
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
# make graphics inline


import warnings
warnings.filterwarnings("ignore")

# get the classnames from the directory structure
# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("/Users/applepei/Documents/Work/Kaggle/Plankton2015","train", "*"))\
 ).difference(set(glob.glob(os.path.join("/Users/applepei/Documents/Work/Kaggle/Plankton2015","","*.*")))))


directory_test = list(set(glob.glob(os.path.join("/Users/applepei/Documents/Work/Kaggle","Plankton2015", "test"))\
 ).difference(set(glob.glob(os.path.join("/Users/applepei/Documents/Work/Kaggle","Plankton2015","test.*")))))

#%% 
#%%
# print directory_test

# Example image
# This example was chosen for because it has two noncontinguous pieces
# that will make the segmentation example more illustrative
test_file = glob.glob(os.path.join(directory_test[0],"*.jpg"))[1819]
# print test_file
im = imread(test_file, as_grey=True)
plt.imshow(im, cmap=cm.gray)
plt.show() 



# First we threshold the image by only taking values greater than the mean to reduce noise in the image
# to use later as a mask
f = plt.figure(figsize=(12,3))
imthr = im.copy()
imthr = np.where(im > np.mean(im),0.,1.0)
sub1 = plt.subplot(1,4,1)
plt.imshow(im, cmap=cm.gray)
sub1.set_title("Original Image")

sub2 = plt.subplot(1,4,2)
plt.imshow(imthr, cmap=cm.gray_r)
sub2.set_title("Thresholded Image")

imdilated = morphology.dilation(imthr, np.ones((4,4)))
sub3 = plt.subplot(1, 4, 3)
plt.imshow(imdilated, cmap=cm.gray_r)
sub3.set_title("Dilated Image")

labels = measure.label(imdilated)
labels = imthr*labels
labels = labels.astype(int)
sub4 = plt.subplot(1, 4, 4)
sub4.set_title("Labeled Image")
plt.imshow(labels)


# calculate common region properties for each region within the segmentation
regions = measure.regionprops(labels)


# find the largest nonzero region
def getLargestRegion(props=regions, labelmap=labels, imagethres=imthr):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop
    
    
f = plt.figure(figsize=(6,3))    
regionmax = getLargestRegion()
plt.imshow(np.where(labels == regionmax.label,1.0,0.0))
plt.show()



print regionmax.minor_axis_length/regionmax.major_axis_length
print regionmax.area
print regionmax.bbox
print regionmax.minor_axis_length,regionmax.major_axis_length
print regionmax.orientation

#%%


#%%
# create function MinorMajorRatio -
def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
    return ratio

#%%
#########
## Preparing Training Data Sets
#########
# Rescale the images and create the combined metrics and training labels

#%%
#get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1

# We'll rescale the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
num_features = imageSize + 1 # for our ratio

#X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X_train = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y_train = np.zeros((num_rows))

files = []
# Generate training data
i = 0    
label = 0
# List of string of class names
namesClasses = list()

print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)
            axisratio = getMinorMajorRatio(image)
            image = resize(image, (maxPixel, maxPixel))
            
            # Store the rescaled image pixels and the axis ratio
            X_train[i, 0:imageSize] = np.reshape(image, (1, imageSize))
            X_train[i, imageSize] = axisratio
            
            # Store the classlabel
            y_train[i] = label
            i += 1
            # report progress for each 5% done  
            report = [int((j+1)*num_rows/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / num_rows), "% done"
    label += 1
    
#%%    


#########
## Preparing Test Data Sets
#########
# Rescale the images and create the combined metrics and training labels
#%%
numberofImages = 0
for folder in directory_test:
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1

# We'll rescale the images to be 25x25
maxPixel = 25
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
num_features = imageSize + 1 # for our ratio

# X is the feature vector with one row of features per image
# consisting of the pixel values and our metric
X = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y = np.zeros((num_rows))

files_submit = []
# Generate training data
i = 0    
label = 0
# List of string of class names
namesClasses = list()

print "Reading images"
# Navigate through the list of directories
for folder in directory_test:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
            image = imread(nameFileImage, as_grey=True)
            files_submit.append(nameFileImage)
            axisratio = getMinorMajorRatio(image)
            image = resize(image, (maxPixel, maxPixel))
            
            # Store the rescaled image pixels and the axis ratio
            X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
            X[i, imageSize] = axisratio            
            
            # Store the classlabel
            y[i] = label
            i += 1
            # report progress for each 5% done  
            report = [int((j+1)*num_rows/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / num_rows), "% done"
    label += 1

  
#%%


#### Visualize test data features
## Loop through the classes two at a time and compare their distributions of the Width/Length Ratio
#
##Create a DataFrame object to make subsetting the data on the class 
#df = pd.DataFrame({"class": y[:], "ratio": X[:, num_features-1]})
#
#f = plt.figure(figsize=(30, 20))
##we suppress zeros and choose a few large classes to better highlight the distributions.
#df = df.loc[df["ratio"] > 0]
#minimumSize = 20 
#counts = df["class"].value_counts()
#largeclasses = [int(x) for x in list(counts.loc[counts > minimumSize].index)]
## Loop through 40 of the classes 
#for j in range(0,40,2):
#    subfig = plt.subplot(4, 5, j/2 +1)
#    # Plot the normalized histograms for two classes
#    classind1 = largeclasses[j]
#    classind2 = largeclasses[j+1]
#    n, bins,p = plt.hist(df.loc[df["class"] == classind1]["ratio"].values,\
#                         alpha=0.5, bins=[x*0.01 for x in range(100)], \
#                         label=namesClasses[classind1].split(os.sep)[-1], normed=1)
#
#    n2, bins,p = plt.hist(df.loc[df["class"] == (classind2)]["ratio"].values,\
#                          alpha=0.5, bins=bins, label=namesClasses[classind2].split(os.sep)[-1],normed=1)
#    subfig.set_ylim([0.,10.])
#    plt.legend(loc='upper right')
#    plt.xlabel("Width/Length Ratio")
#
    
#%%
#np.savetxt("/Users/applepei/Documents/Work/Kaggle/Plankton2015/TutorialTest_X.csv", X, delimiter=",")
#np.savetxt("/Users/applepei/Documents/Work/Kaggle/Plankton2015/TutorialTest_X_files.csv", files_submit, delimiter=",", fmt="%s")
    
X=np.genfromtxt("/Users/applepei/Documents/Work/Kaggle/Plankton2015/TutorialTest_X.csv",delimiter=',')

files_submit=np.genfromtxt("/Users/applepei/Documents/Work/Kaggle/Plankton2015/TutorialTest_X_files.csv",delimiter=',',dtype=str)
#%%

#%%
#import pandas as pd
pd3 = pd.read_csv("/Users/applepei/Documents/Work/Kaggle/Plankton2015/sampleSubmission.csv",header=0)

print pd3.columns[1:100]
print y_train
y_train_label = []
for i in range(0,len(files)):
    y_train_label.append(files[i].split('/')[8])

column_head = pd3.columns[1:]

align_y_train = np.zeros(len(y_train))
for j in range(0,len(y_train)):
    for k in range(0,len(column_head)):
        if y_train_label[j]==column_head[k]:
            align_y_train[j]=k
#%%


#%%
## PCA / LDA Practice
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.lda import LDA

print align_y_train[1:100]

pca = PCA()
X_r = pca.fit(X_train).transform(X_train)

lda = LDA()
X_r2 = lda.fit(X_train,align_y_train).transform(X_train)
#%%

#%%
print len(X_train[0])
print len(X_r2[0])

y_pred = lda.predict(X)
y_pred_v = lda.predict(X_train)
#%%

#%%
print align_y_train[11010:11055],y_train_label[11010:11055],y_pred_v[11010:11055]
print pd3.columns[73]
#%%
num_rows = len(y_pred)
jpg_name=[]
for k in range(0,num_rows):
    jpg_name.append(files_submit[k].split('/')[8])
print y_pred,len(y_pred)
#%%

#%%
num_rows = len(y_pred) # one row for each image in the training dataset
num_classes = 122

test_perf =[]
for r in range(0,num_rows):
    row_submit = []
    for i in range(0,num_classes):
        if i==0:
            row_submit.append(jpg_name[r])
        else:
            if i==y_pred[r]:
                row_submit.append(0.99173554)
            else:
                row_submit.append(0.00826446)
    
    test_perf.append(row_submit)

pd3.loc[0:num_rows-1]=test_perf[:]

#%%
  

#%%

print pd3.columns[0]

#print pd1.ix[num_rows-1]

#%%

#kf = KFold(y, n_folds=5)
#print kf

#y_pred = y * 0
#for train, test in kf:
#    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
#    clf = RF(n_estimators=100, n_jobs=3)
#    clf.fit(X_train, y_train)
#    y_pred[test] = clf.predict(X_test)

# print classification_report(y, y_pred, target_names=namesClasses)



#%%
pd3.to_csv("/Users/applepei/Documents/Work/Kaggle/Plankton2015/pd3Submission.csv", index = False)
#%%
