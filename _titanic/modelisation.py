from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

##Logistic Regression
def Logistic_Regression_Model (X,y,size,RdomState=42):
    #X,y
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size,random_state=RdomState)
    model=LogisticRegression(random_state=RdomState)
    model.fit(X_train,y_train)
    #run the model
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1]
    score_train=model.score(X_train,y_train)
    score_test=model.score(X_test,y_test)
    return {"y_test":y_test,"prediction":y_pred,"proba":y_prob,"score_train":score_train,"score_test":score_test,"model":model}

##Random Forest
def Random_Forest_Model (X,y,size,RdomState=42):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size,random_state=RdomState)
    model=RandomForestClassifier(random_state=42,n_estimators=100,criterion="gini",max_depth=20)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1]
    score_train=model.score(X_train,y_train)
    score_test=model.score(X_test,y_test)
    return {"y_test":y_test,"prediction":y_pred,"proba":y_prob,"score_train":score_train,"score_test":score_test,"model":model}

##GridSearchCV
def GridSearchCV_Model (X,y,size,RdomState=42):
    Estimator = RandomForestClassifier(random_state=42)
    parameters={
    'n_estimators':[100,150,200,250,300],
    'max_depth': np.arange(6,16,2),
    'min_samples_split': np.arange(10,30,5),
    'min_samples_leaf' : np.arange(5,20,5)
    }
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size,random_state=RdomState)
    model=GridSearchCV(Estimator,parameters,verbose=1,cv=5,n_jobs=-1)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1]
    score_train=model.score(X_train,y_train)
    score_test=model.score(X_test,y_test)
    return {"y_test":y_test,"prediction":y_pred,"proba":y_prob,"score_train":score_train,"score_test":score_test,"model":model}
