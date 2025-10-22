"""

Rev for general basic Machine Learning ..
-- Supervised Learning & Unsupervised Learning

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier ,plot_tree 
from sklearn.model_selection import train_test_split ,GridSearchCV,ParameterGrid ,cross_val_score 
from sklearn.preprocessing import PolynomialFeatures ,StandardScaler
from sklearn.cluster import KMeans ,AgglomerativeClustering
from scipy.cluster.hierarchy import linkage,dendrogram ,fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN




"""
np.random.seed(42)
size = np.random.randint(600, 2500, 40)
price = 50000 + size * 120 + np.random.randint(-15000, 15000, 40)

df = pd.DataFrame({'size': size, 'price': price})
df.head()
x=df[['size']].values
y=df['price'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model= LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

plt.plot(x,model.predict(x),c='pink')
plt.show()

print('-------------Seperate---------------') 


np.random.seed(42)
area = np.random.randint(50, 200, 40)
rooms = np.random.randint(1, 5, 40)
rent = 1000 + area * 20 + rooms * 500 + np.random.randint(-300, 300, 40)

df = pd.DataFrame({'area': area, 'rooms': rooms, 'rent': rent})
df.head()

x=df[['area','rooms']].values
y=df['rent'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(model.coef_) 

#important feature is rooms 

plt.figure(figsize=(9,6))
plt.subplot(1,2,1)
plt.scatter(df['area'],df['rent'],c='blue',label='Area') 
plt.scatter(df['rooms'],df['rent'],c='red',label='Rooms') 
plt.title('True Data')
plt.subplot(1,2,2)
plt.plot(df[['area','rooms']],model.predict(df[['area','rooms']].assign(area=3)))
plt.title('Prediction')
plt.tight_layout()
plt.show()

print('-------------Seperate---------------') 


np.random.seed(42)
exp = np.linspace(0, 20, 40)
salary = 2500 + 500 * exp + 50 * (exp ** 2) + np.random.randint(-1000, 1000, 40)

df = pd.DataFrame({'experience': exp, 'salary': salary})
df.head()

x=df[['experience']].values
y=df['salary'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#----

poly=PolynomialFeatures(degree=2,include_bias=False)
x_TR=poly.fit_transform(x_train)
x_TS=poly.transform(x_test)

model_p=LinearRegression()
model_p.fit(x_TR,y_train)
y_pp=model_p.predict(x_TS)
new=np.array([[15]])
new_p=poly.transform(new)

print("predicted salary for exp= 15 => ",round(model_p.predict(new_p)[0],2))

x_poly=poly.transform(x)
y_poly=model_p.predict(x_poly)


plt.figure(figsize=(9,6))
plt.subplot(1,2,1)
plt.plot(x,model.predict(x),c='blue')
plt.subplot(1,2,2)
plt.plot(x_poly,y_poly)
plt.show()

print('-------------Seperate---------------') 







##  Logistic Regression → (study hours → pass/fail)


np.random.seed(42)
hours = np.random.randint(1, 10, 40)
pass_exam = (hours > 5).astype(int)

df = pd.DataFrame({'hours': hours, 'pass_exam': pass_exam})
df.head()

x=df[['hours']].values
y=df['pass_exam'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=LogisticRegression(max_iter=200)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

x_curve=np.linspace(max(x),min(x),40)
y_prob=clf.predict_proba(x_curve)

plt.subplot(1,3,1)
plt.scatter(x,y,c='pink')
plt.subplot(1,3,2)
plt.plot(x,clf.predict(x))
plt.subplot(1,3,3)
plt.plot(x_curve,y_prob)
plt.show()

print('-------------Seperate---------------') 




##  Logistic Regression (2 features) → (GPA + activities → accepted)


np.random.seed(42)
GPA = np.round(np.random.uniform(2, 4, 40), 2)
activities = np.random.randint(0, 10, 40)
accepted = ((GPA > 3) & (activities > 4)).astype(int)

df = pd.DataFrame({'GPA': GPA, 'activities': activities, 'accepted': accepted})
df.head()

x=df[['GPA','activities']].values
y=df['accepted'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

clf=LogisticRegression(max_iter=200)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

acc=clf.score(x_test,y_test)
print('-- Accuracy => ',acc)
y_prob=clf.predict_proba(x_test)[:,1]

x_curve=np.linspace(x.max(),x.min(),40).reshape(-1,1)
y_prob_plt=clf.predict_proba(df[['GPA','activities']].assign(activities=1))[:,1]
plt.plot(x_curve,y_prob_plt)
plt.show()

print(clf.coef_)
#important feature is GPA

print('-------------Seperate---------------') 



##  Decision Tree Classifier → (Titanic-like dataset)


np.random.seed(42)
age = np.random.randint(1, 80, 50)
sex = np.random.choice(['male', 'female'], 50)
fare = np.round(np.random.uniform(5, 200, 50), 2)
survived = ((fare > 50) & (sex == 'female')).astype(int)

df = pd.DataFrame({'age': age, 'sex': sex, 'fare': fare, 'survived': survived})
df.head()
df['sex']=df['sex'].map({'male':0,'female':1})

x=df[['age','sex','fare']].values
y=df['survived'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
tree=DecisionTreeClassifier(random_state=42)
params={
    'max_depth' :[2,3,4,5],
    'min_samples_split':[2,3,4,5] ,
    'criterion':['gini','entropy'] ,

}
grid=GridSearchCV(tree,param_grid=params,cv=3,scoring='accuracy')
grid.fit(x_train,y_train)
y_pred=grid.best_estimator_.predict(x_test)
print(grid.best_params_)

plot_tree(grid.best_estimator_,max_depth=2,feature_names=df[['age','sex','fare']].values,class_names=['Not Survived','Survived'],filled=True)
plt.show()

print('-------------Seperate---------------') 



## Metrics (MSE, MAE, RMSE, R²)


np.random.seed(42)
true = np.random.randint(100, 300, 20)
pred = true + np.random.randint(-30, 30, 20)

df = pd.DataFrame({'True': true, 'Predicted': pred})
df.head()



## Metrics (confusion matrix, precision, recall, F1, ROC)


np.random.seed(42)
true = np.random.randint(0, 2, 30)
pred = true.copy()
noise_idx = np.random.choice(range(30), 6, replace=False)
pred[noise_idx] = 1 - pred[noise_idx]

df = pd.DataFrame({'True': true, 'Predicted': pred})
df.head()


##  Train/Test Split + Cross-validation


np.random.seed(42)
x1 = np.random.randn(100)
x2 = np.random.randn(100)
y = (3*x1 - 2*x2 + np.random.randn(100)*0.5)

df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
df.head()

x=df[['x1','x2']].values
y=df['y'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc=model.score(x_test,y_test)
print('-- Accuracy With Split => ',acc)

model_r=LinearRegression()
cross=cross_val_score(model_r,x,y,cv=5,scoring='r2')
print('-- Accuracy with Cross validation => ',cross)

print('-------------Seperate---------------') 

##  GridSearchCV + Hyperparameter tuning (Decision Tree)


np.random.seed(42)
x1 = np.random.randint(0, 10, 60)
x2 = np.random.randint(0, 10, 60)
y = ((x1 + x2) > 10).astype(int)

df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
df.head()

x=df[['x1','x2']].values
y=df['y'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)

tree=DecisionTreeClassifier(random_state=42)
params={
    'max_depth':[2,3,4,5],
    'min_samples_split':[2,3,4,5],
    'criterion':['gini','entropy'],

}

grid=GridSearchCV(tree,param_grid=params,cv=3,scoring='accuracy')
grid.fit(x_train,y_train)
y_pred=grid.best_estimator_.predict(x_test)
print(grid.best_params_)

plot_tree(grid.best_estimator_,max_depth=2,feature_names=df[['x1','x2']].values,class_names=['Zero','One'])
plt.show()






## KMeans Clustering (customer segmentation)


from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)
df = pd.DataFrame(x, columns=['income', 'spending_score'])
df.head()

scale=StandardScaler()
x_scaled=scale.fit_transform(x)

inertia=[]
K=range(1,11)
for k in K:
    kmean=KMeans(n_clusters=k,random_state=42)
    kmean.fit(x_scaled)
    inertia.append(kmean.inertia_)

plt.scatter(K,inertia)
plt.show()

kmean=KMeans(n_clusters=2,random_state=42)
kmean.fit(x_scaled)
df['cluster']=kmean.labels_

plt.scatter(x_scaled[:,0],x_scaled[:,1],c=df['cluster'],cmap='coolwarm',alpha=0.8,s=50)
plt.show()




## Hierarchical Clustering (dendrogram)


from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)
df = pd.DataFrame(x, columns=['feature1', 'feature2'])
df.head()

x=df[['feature1','feature2']].values

scale=StandardScaler()
x_scaled=scale.fit_transform(x)

z=linkage(x_scaled,method='ward')

plt.figure(figsize=(10,6))
dend=dendrogram(z,orientation='top'
                ,show_leaf_counts=True
                ,distance_sort='ascending')
plt.axhline(y=12,linestyle='--',c='r')
plt.show()

cluster_from_dend=fcluster(z,t=12,criterion='distance')
df['cluster_from_dend']=cluster_from_dend

#print(df.groupby('cluster_from_dend')['feature1','feature2'].mean())
print(df['cluster_from_dend'].value_counts())

#------------
agg=AgglomerativeClustering(n_clusters=12,metric='euclidean',linkage='ward')
agg_label=agg.fit_predict(x_scaled)
df['cluster_agg']=agg_label
print(df['cluster_agg'].value_counts())

plt.scatter(x_scaled[:,0],x_scaled[:,1],c=df['cluster_agg'],cmap='tab10',alpha=0.8,s=50)
plt.show()



##  DBSCAN Clustering → detect outliers

from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
outliers = np.random.uniform(low=-6, high=6, size=(10, 2))
X_full = np.vstack([X, outliers])
df = pd.DataFrame(X_full, columns=['x', 'y'])
df.head()

x=df[['x','y']].values
scale=StandardScaler()
x_scaled=scale.fit_transform(x)

dbscn=DBSCAN(eps=0.5,min_samples=5)
df['cluster'] = dbscn.fit_predict(x_scaled)

plt.scatter(x_scaled[:,0],x_scaled[:,1],c=df['cluster'],cmap='coolwarm',alpha=0.8,s=50)
plt.show()



## PCA (Dimensionality Reduction)


np.random.seed(42)
X1 = np.random.randn(100) * 2 + 10
X2 = X1 * 0.5 + np.random.randn(100) * 0.5
X3 = X1 * -0.2 + X2 * 0.4 + np.random.randn(100) * 0.3
df = pd.DataFrame({'x1': X1, 'x2': X2, 'x3': X3})
df.head()
x=df[['x1','x2','x3']].values
scale=StandardScaler()
x_scaled=scale.fit_transform(x)

pca=PCA(n_components=None,random_state=42)
pca.fit(x_scaled)

var_ratio=pca.explained_variance_ratio_

cummulative=np.cumsum(var_ratio)

plt.subplot(1,2,1)
plt.bar(range(1,len(var_ratio)+1),var_ratio)
plt.title('Basic Data')
plt.subplot(1,2,2)
plt.plot(range(1,len(cummulative)+1),cummulative)
plt.title('Compnents Need !')
plt.axhline(y=0.97,linestyle='--',c='r')
plt.tight_layout()
plt.show()

n=np.argmax(cummulative >= 0.79) +1
pca=PCA(n_components=n,random_state=42)
xpc=pca.fit_transform(x_scaled)
sc=plt.scatter(xpc[:,0],xpc[:,1],cmap='rainbow',alpha=0.8,s=50)
plt.colorbar(sc,label='target')
plt.show()

"""



##  t-SNE visualization (MNIST small or synthetic)

from sklearn.datasets import load_digits
digits = load_digits()
x = digits.data
y = digits.target

scale=StandardScaler()
x_scaled=scale.fit_transform(x)

perp = [5, 20, 50, 100]
plt.figure(figsize=(12,10))
for i,p in enumerate(perp):
  
  
  tsne=TSNE(n_components=2,perplexity=p,learning_rate=200,random_state=42)
  xtsn=tsne.fit_transform(x_scaled)

  plt.subplot(2,2,i+1) 
  plt.scatter(xtsn[:,0],xtsn[:,1],c=y,cmap='rainbow',alpha=0.8,s=50)
  plt.title(f"when Perp = {p}")
plt.tight_layout()
plt.show()
