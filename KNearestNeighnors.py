#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#SEABORN STYLE
sns.set_style('whitegrid')

#READ CSV
df = pd.read_csv('C:\\Users\\User\\Documents\\KNN_Project_Data')

#Change Working Directory if required
import os
os.getcwd()
os.chdir('C:\\Users\\User\\Desktop\\school\\Python\\projects\\KNN')

#EDA
print('\n',df.head(),'\n')

print('\n',df.info(),'\n')

print('\n',df.describe(),'\n')

i1 = sns.pairplot(data=df,hue='TARGET CLASS',diag_kind='hist',palette='coolwarm')
i1.savefig('PairPlot.jpg')
plt.show()

#The datapoints are not in scale, we need to standardize
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df[df.columns[:-1]])

scaled_data = scaler.transform(df[df.columns[:-1]])

scaled_variables = pd.DataFrame(data=scaled_data,columns=[df.columns[:-1]])

print('Scaled Variable: ','\n')
print('\n',scaled_variables.head(),'\n')

#Train Test Split
from sklearn.model_selection import train_test_split

X = scaled_variables
y= df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#KNN, k=1 (To test performance)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

#Predictions and Evaluations
predictions = knn.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix

print('Confusion Matrix: \n',confusion_matrix(y_test,predictions),'\n')

print('Classification Report: \n',classification_report(y_test,predictions),'\n')

sns.set_style('white')
i2 = plot_confusion_matrix(knn,X_test,y_test)
plt.savefig('Confusion Matrix,k=1.jpg')
plt.show()


#Choosing K Value

#FOR LOOP
error_rate = []

for i in range(1,80):
    
    knn=KNeighborsClassifier(n_neighbors=i)
    
    knn.fit(X_train,y_train)
    
    pred_i = knn.predict(X_test)
    
    error_rate.append(np.mean(pred_i != y_test))

#K versus Error Rate Plot
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.plot(range(1,80),error_rate,color='blue',marker='o',markerfacecolor='red',markersize=10)
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.title('K vs. Error Rate')
plt.tight_layout()
plt.savefig('K vs. Error.jpg')
plt.show()


#BEST K= 31
# KNN with k=31
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train,y_train)
preds = knn.predict(X_test)
print('WITH K = 31\n')
print('Confusion Matrix: \n',confusion_matrix(y_test,preds),'\n')
print('Classification Report: \n',classification_report(y_test,preds),'\n')

sns.set_style('white')
plot_confusion_matrix(knn,X_test,y_test)
plt.savefig('Confusion Matrix,k=31.jpg')
plt.show()


