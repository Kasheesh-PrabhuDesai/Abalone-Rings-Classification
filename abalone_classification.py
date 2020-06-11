import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold,GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from math import sqrt
from sklearn.feature_selection import RFE

get_ipython().run_line_magic('matplotlib','qt')
names=['Sex','Length','Diameter','Height','Whole Weight','Shuckered Weight','Viscera Weight','Shell Weight','Rings']

df = pd.read_csv('abalone-data.csv',names=names)

df.hist()
df.plot(kind='density',subplots=True,layout=(3,3),sharex=False,sharey=False,fontsize=10)

df.plot(kind='box',subplots=True,layout=(3,3),sharex=False,sharey=False,fontsize=10)

print(df.corr(method='pearson'))

correlations = df.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

array = df.values
x = array[:,:-1]
y = array[:,-1].astype('float64')

oe = OrdinalEncoder()
new_x = oe.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(new_x,y,test_size=0.20,random_state=7)

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

results = []


names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print('\t')    
    
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',
Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',
ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print('\t')


model = LinearRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(sqrt(mean_squared_error(y_test,predictions)))


print('\t')

model = Lasso()
alpha = [0.2,0.4,0.6,0.8,1,2,3,4,5,6,7,8]
param_grid = dict(alpha=alpha)
kfold = KFold(n_splits=10,random_state=7,shuffle=True)
grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=kfold,scoring='neg_mean_squared_error')
grid_result = grid.fit(x_train,y_train)
print(grid_result.best_score_,grid_result.best_params_)


model = ElasticNet(alpha=0.2)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(sqrt(mean_squared_error(y_test,predictions)))
