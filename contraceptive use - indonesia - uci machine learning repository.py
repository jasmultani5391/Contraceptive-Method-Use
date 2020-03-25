#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


import matplotlib.pyplot as plt

import numpy as np
from numpy import array
from numpy import argmax

data = pd.read_csv('C:\\Users\Jasmine\Downloads\cmc.data', sep=',')
data.columns = [
    'wife_age', 
    'wife_edu', 
    'husband_edu', 
    '#kids', 
    'wife_religion', 
    'wife_working?', 
    'husb_job', 
    'standard_of_living', 
    'media_exposure', 
    'bcontrol']
print(data.head(10))


### info on the features, from UCI machine learning repository
### 
# 1. Wife's age (numerical)
# 2. Wife's education (categorical) 1=low, 2, 3, 4=high
# 3. Husband's education (categorical) 1=low, 2, 3, 4=high
# 4. Number of children ever born (numerical)
# 5. Wife's religion (binary) 0=Non-Islam, 1=Islam
# 6. Wife's now working? (binary) 0=Yes, 1=No
# 7. Husband's occupation (categorical) 1, 2, 3, 4
# 8. Standard-of-living index (categorical) 1=low, 2, 3, 4=high
# 9. Media exposure (binary) 0=Good, 1=Not good
# 10. Contraceptive method used (class attribute) 1=No-use, 2=Long-term, 3=Short-term





sns.countplot(y = 'wife_age', hue = 'bcontrol', data = data)


#higher the number ~ higher the wife's education
#volume of birth control increases by education but doesn't surpass "not using bcontrol" until the highest education
#sns.countplot(y = 'wife_edu', hue = 'bcontrol', data = data)

#wowowow, not using bc exceeds each category
#sns.countplot(y = 'husband_edu', hue = 'bcontrol', data = data)

#general trend of trying to start a family, 0 = most likely to not have bcontrol
#sns.countplot(y = '#kids', hue = 'bcontrol', data = data)

#wife's religion 0 = nonIslam, 1 = Islam
#major change---all choices have equal weight under non-religious category whereas Islam has some preferences
#sns.countplot(y = 'wife_religion', hue = 'bcontrol', data = data)

#wife working = 1, wife not working = 0
#volume increases from "wife working" to "not working", but the general choice to stay without birth control wins in each
#sns.countplot(y = 'wife_working?', hue = 'bcontrol', data = data)

#sns.countplot(y = 'husb_job', hue = 'bcontrol', data = data)

#volume to use bc increases with higher standard of living, but choice to do use bc wins
#sns.countplot(y = 'standard_of_living', hue = 'bcontrol', data = data)

# 0 = good media coverage, 1 = not good
#sns.countplot(y = 'media_exposure', hue = 'bcontrol', data = data)


sns.lmplot(x='wife_age', y='#kids', data=data,
           fit_reg=False, # No regression line
           hue= 'bcontrol')   # Color by birth control

#above shows a really nice trend
#those who aren't on birth control, are often the ones with 0 or 1 kid (probably to start a family)
#I notice that there's another clump of women (around age 40, with 10+ kids) who also don't have use birth control
#data.describe()

sns.lmplot(x='wife_age', y='wife_edu', data=data,
           fit_reg=False, # No regression line
           hue= 'bcontrol')   # Color by birth control



# In[144]:


#prepping the data so that numbers are normalized; normalize data that isn't already in a binary code (wife_age, wife_edu, husband_edu< #kids, husb_job, standard_of_living)


class Cleanup():
    def normdata(self, datapoint):
        minimum = min(datapoint)
        maximum = max(datapoint)
        normalized = []
        for i in datapoint:
            normalized += [(i-minimum)/(maximum-minimum)]
        return normalized
    
    def countinglabels(self, my_list):
        freq = {}
        for i in my_list:
            if (i in freq):
                freq[i] += 1
            else:
                freq[i] = 1
        return freq
    
    def label(self, data):
        lbl = []
        for i in data:
            if i == 1:
                lbl.append(0)
            else:
                lbl.append(1)
        return lbl
        

wifeage = Cleanup()
data['wife_age_norm'] = wifeage.normdata(data['wife_age'])

wifeedu = Cleanup()
data['wife_edu_norm'] = wifeedu.normdata(data['wife_edu'])

husbedu = Cleanup()
data['husband_edu_norm'] = husbedu.normdata(data['husband_edu'])

husbjob = Cleanup()
data['husb_job_norm'] = husbjob.normdata(data['husb_job'])

numkids = Cleanup()
data['#kids_norm'] = husbjob.normdata(data['#kids'])

standardofliving = Cleanup()
data['#standard_of_living_norm'] = husbjob.normdata(data['standard_of_living'])


# also want to simplify the target some more; let's make 0 = no birth control at all (previously 1 in raw data),
# and 1 = some or highest form of data (previously 2 and 3 in raw data, respectively)

list_bcontrol = list(data['bcontrol'])
binary_bc = Cleanup()
binary_bcontrol = binary_bc.label(list_bcontrol)


#below helps us double check that the lengh of the raw data's birth control column is equal to the binary birth control label we created
listbcontrol = Cleanup()
listbcontrol = listbcontrol.countinglabels(list_bcontrol)
print(listbcontrol)

binbcontrol = Cleanup()
binbcontrol = binbcontrol.countinglabels(binary_bcontrol)
print(binbcontrol)

#print(333+511), this equals 844


# In[153]:


#Feature engineering: let's decide which features to include for our future machine learning algorithm


features = pd.DataFrame(
    {'wifeage_norm': data['wife_age_norm'],
     'wifeedu_norm': data['wife_edu_norm'],
     'husbandedu_norm': data['husband_edu_norm'],
     'kids_norm': data['#kids_norm'],
     'wife_religion' : data['wife_religion'],
     'wife_working?' : data['wife_working?'],
     'husbjob_norm' : data['husb_job_norm'],
     'standardofliving_norm': data['#standard_of_living_norm'],
     'media_exposure': data['media_exposure']
    })

label = pd.DataFrame({'label' : binary_bcontrol})
multilabel = pd.DataFrame({'label' : data['bcontrol']})
data['binary label'] = binary_bcontrol


feat_label = [features, label]



print(data.head(5))


# In[146]:


#logistic regression algorithm

traindata, testdata, trainlbl, testlbl = train_test_split(features,label,test_size = 0.2, random_state = 3)

scaler = StandardScaler()
traindata = scaler.fit_transform(traindata)
testdata = scaler.transform(testdata)

model = LogisticRegression()
model.fit(traindata, np.ravel(trainlbl))


# Score the model on the train data
trainmodel = model.score(traindata, trainlbl) #should parameters be the same as .fit()
print('train score is ' + str(round(trainmodel*100,2)) + ' %')
# Score the model on the test data
testmodel = model.score(testdata, testlbl)
print('test score is ' + str(round(testmodel*100)) + ' %')


# In[148]:


#decision forest

traindataF, testdataF, trainlblF, testlblF = train_test_split(features, label, random_state = 3) #random-state is sqrt of number of total features from raw data


forest = RandomForestClassifier(random_state=3)
forest.fit(traindataF, np.ravel(trainlblF))
print('Accuracy of Forest classifier is ' + str(forest.score(testdataF, testlblF)*100) + ' %')
weights_features = forest.feature_importances_
columns = features.head(0)
variable_weights = list(zip(columns, weights_features))
print('\t')
print(variable_weights)
biggest_index = max(weights_features)
print('\t')
print(biggest_index) #this states that wife_age_norm and kids_norm are the leading two variables that influence correlation with label


# In[154]:


#evaluating features

# accuracy was previously 69.8369% for our Forest Classifier...which means
# it was about 69.83 % correct ((true positive+true negative)/total)




#using info from Forest classifier
scores = []

predictlabel= forest.predict(testdataF)
precisionForest = precision_score(testlblF, predictlabel, average = 'binary')
recallForest = recall_score(testlblF, predictlabel, average='binary')
F1Forest = f1_score(testlblF, predictlabel, average='binary')

scores.append(precisionForest*100)
scores.append(recallForest*100)
scores.append(F1Forest*100)

scorenames = ['precision %', 'recall %', 'F1 %']
ForestScores = list(zip(scorenames, scores))

print(ForestScores)


sns.lmplot(x='wife_age', y='#kids', data=data,
           fit_reg=False, # No regression line
           hue= 'binary label')   # Color by birth control



# when our model predicts positive, it was 73.54 % correct
# precision helps when cost of false positive is high (the woman is reported to be taking birth control, but actually isn't)
# recall helps with cost of false negative is high (the woman is stated to not use birth control, but actually is using it)
# F1 score is an overall measure of model's accuracy (combines precision and recall)...(2* (precisions * recall)/(precision+recall)); closer to 1 the better


# In[150]:


## instead of using labels as a binary (uses birth control or not), will set up the decision forest classification to adjust for multiclass labels

traindataFII, testdataFII, trainlblFII, testlblFII = train_test_split(features, multilabel, random_state = 3) #random-state is sqrt of number of total features from raw data


forestII = RandomForestClassifier(random_state=3)
forestII.fit(traindataFII, np.ravel(trainlblFII))
print('Accuracy of Forest classifier is ' + str(forestII.score(testdataFII, testlblFII)*100) + ' %')
weights_featuresII = forestII.feature_importances_
columns = features.head(0)
variable_weightsII = list(zip(columns, weights_featuresII))
print("\t")
print(variable_weightsII)
biggest_indexII = max(weights_featuresII)
#print(biggest_indexII)

#performance drops terribly...maybe I need to use a different algorithm to address multi-classification?


scoresII = []

predictlabelII= forestII.predict(testdataFII)
precisionForestII = precision_score(testlblFII, predictlabelII, average = 'weighted')
recallForestII = recall_score(testlblFII, predictlabelII, average='weighted')
F1ForestII = f1_score(testlblFII, predictlabelII, average='weighted')

scores.append(precisionForestII*100)
scores.append(recallForestII*100)
scores.append(F1ForestII*100)

scorenames = ['precision %', 'recall %', 'F1 %']
ForestScoresII = list(zip(scorenames, scoresII))

print("\t")
print('precision %: ' + str(precisionForestII*100), 'recall %: ' + str(recallForestII*100), 'F1 %: ' + str(F1ForestII*100))
#print(ForestScoresII)


# In[151]:


#will try to use K-nearest neighbor for multi-class use
# ....this is even worse

traindataFII, testdataFII, trainlblFII, testlblFII = train_test_split(features, multilabel, random_state = 3) #random-state is sqrt of number of total features from raw data

KNN = KNeighborsClassifier(n_neighbors = 48)
KNNfit = KNN.fit(traindataFII, np.ravel(trainlblFII))
KNNAcc = KNN.score(testdataFII, testlblFII)

scores = []

predictlabelKNN= KNN.predict(testdataFII)
precisionKNN = precision_score(testlblFII, predictlabelKNN, average='weighted')
recallKNN = recall_score(testlblFII, predictlabelKNN, average='weighted')
F1KNN = f1_score(testlblFII, predictlabelKNN, average='weighted')

scores.append(precisionKNN*100)
scores.append(recallKNN*100)
scores.append(F1KNN*100)

scorenames = ['precision', 'recall', 'F1']
KNNscores = list(zip(scorenames, scores))
print("Accuracy is: " + str(KNNAcc*100))
print("Scores for KNN are: " + str(KNNscores))


# In[ ]:




