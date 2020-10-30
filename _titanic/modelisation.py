##Logistic Regression
def My_model (X,y,size,RdomState=42):
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
