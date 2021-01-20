from config import *
# baseline model
def create_baseline():
	# create model
	np.random.seed(7)
	model = Sequential()
	model.add(Dense(12, input_dim=number_of_dimensions, activation='relu'))
	model.add(Dense(5, activation='relu'))

	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_


    return grid_search.best_params_

# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k=5)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs


def visualize_DecisionTree(clf,path,features):
    dot_data = export_graphviz(clf, out_file=None,feature_names=features) 
    graph = graphviz.Source(dot_data) 
    graph.render(path+"Decision_Tree") 

def plotData(x,target,figurepath=""):
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])
    
    finalDf = pd.concat([principalDf, target], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1, projection='3d') 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ["'recurrence-events'","'no-recurrence-events'"]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Class'] == target

        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2'],
               finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
    print(pca.explained_variance_ratio_)
    ax.legend(targets,loc='upper right', bbox_to_anchor=(1.09,1.09))
    ax.grid()

    plt.savefig(figurepath)
    plt.clf()
    return finalDf

def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    encoder=LabelEncoder()
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)

    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

def confusionmtrix_tocsv(y_test,y_pred,ctype,path):
	mtx=confusion_matrix(y_test,y_pred)
	pd.DataFrame(mtx).to_csv(path+ctype+"confusion_matrix.csv")
	classrep=classification_report(y_test,y_pred,output_dict=True)
	pd.DataFrame(classrep).transpose().to_csv(path+ctype+"classreport.csv")


def preprocessData(breast_cancer,path):


    #read the files 
    breastcancerDF=pd.read_csv(breast_cancer,skiprows=105,skipfooter=3,header=None,engine='python',names=["age",
    "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig","breast","breast-quad", "irradiat","Class"])

    breastcancerDF.replace("?",np.nan,inplace=True)


    #list of columns to encode
    encodeCols=["age","menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig","breast","breast-quad", "irradiat"]
    x_columns=["age","menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig","breast","breast-quad", "irradiat"]
    y_column=["Class"]
    #encode all column values except the "deg-malig" column
    for column in encodeCols:
    	encode(breastcancerDF[column])

    breastcancerDF.to_csv(path+"preprocessing/after-encoding.csv")
    ####################Drop Null values############################
    breastcancerDF.dropna(subset=['node-caps'], inplace=True)
    breastcancerDF.dropna(subset=['breast-quad'], inplace=True)

    #################################################################

    ######################SET X AND Y SETS COLUMNS#########################
    x_values=breastcancerDF.loc[:,"age":"irradiat"]
    y_values=breastcancerDF.loc[:,"Class"]
    #################################################################


    ####################Null Values Imputing using KNN############################    
    # imputer = KNNImputer(n_neighbors=3,weights='distance')
    # breastcancerDF=pd.DataFrame(np.round(imputer.fit_transform(x_values,y_values)),columns=x_columns)
    # breastcancerDF.to_csv(path+"preprocessing/after-imputing.csv")
    # ##updating the x_values after imputing
    # x_values=breastcancerDF.loc[:,"age":"irradiat"]
    ###############################################################################

    ############################Train/test split #######################################

    x_train,x_test,y_train,y_test=train_test_split(x_values,y_values,test_size=0.34,random_state=24)
    #################################################################################
    
    #####################Feature Selection #####################    
    # x_train, x_test, fs = select_features(x_train, y_train, x_test)


    # # what are scores for the features
    # for i in range(len(fs.scores_)):
    #     print('Feature %d: %f' % (i, fs.scores_[i]))
    # # plot the scores
    # plt.figure(figsize=(20,15))
    # plt.bar(x_columns, fs.scores_)
    # plt.savefig(path+"preprocessing/Feature-selection-5attributes.png")

    # plt.clf()
    # #get the new features names
    # new_features=[]
    # for bool, feature in zip(fs.get_support(), x_columns):
    #     if bool:
    #         new_features.append(feature)
    
    # x_columns=new_features
    ###############################################################

    #####################OverSampling#####################
    
    ##scatter plot of examples by class label
    plotData(x_train,y_train,path+"preprocessing/dataplot-beforeOverSampling.png")
    f = open(path+"preprocessing/classShapeBeforeOverSampling.txt", "a")
    f.write("Class Shape Before Over Sampling")
    f.write(str(Counter(y_train)))
    f.close()

    # In this example I use SMOTENC for oversampling continous and categorical data
    oversampling = SMOTENC(categorical_features=[0,1,2,4],random_state=40)
    x_train, y_train = oversampling.fit_resample(x_train, y_train)

    f = open(path+"preprocessing/classShapeBAfterOverSampling.txt", "a")
    f.write("Class Shape After Over Sampling")
    f.write(str(Counter(y_train)))
    f.close()
    #############################################################

    #################Write Train and test sets##############################

    pd.concat([pd.DataFrame(x_train,columns=x_columns),pd.DataFrame(y_train,columns=y_column)],axis=1).to_csv(path+"preprocessing/train.csv",index=False)
    pd.DataFrame(x_test,columns=x_columns).to_csv(path+"preprocessing/test.csv",index=False)


    plotData(pd.DataFrame(x_train,columns=x_columns),pd.DataFrame(y_train,columns=y_column),path+"preprocessing/dataplot-afterOverSampling.png")

    ###########################################################################
    #####################Decision Tree#####################
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(random_state=40,criterion="entropy",max_depth=tree_max_depth)

    # Train Decision Tree Classifer
    clf = clf.fit(x_train, y_train)
    #visualize DT
    visualize_DecisionTree(clf,path,x_columns)
    #save model for later use
    filename = path+'DecisionTree.sav'
    pickle.dump(clf, open(filename, 'wb'))
    #Predict the response for test dataset
    y_pred = clf.predict(x_test)
    confusionmtrix_tocsv(y_test,y_pred,"DT_",path)
    #########################################################


    #####################SVM#####################
    svm_param=svc_param_selection(x_train, y_train, 25)
    svclassifier = svm.SVC(gamma=svm_param['gamma'],C=svm_param['C'])
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)
    confusionmtrix_tocsv(y_test,y_pred,"svm_",path)


    ############################################

    #####################K Nearest Neighbor#####################
    # instantiate learning model with k=3
    knn = KNeighborsClassifier(n_neighbors = 5)
    # fitting model
    knn.fit(x_train,y_train)
    #predict
    y_pred = knn.predict(x_test)
    confusionmtrix_tocsv(y_test,y_pred,"knn_",path)
    ################################################################


    #####################NEURAL NETWORK#####################    
    #Reshape y so that it can be used in the neural network
    y_train=pd.Series(y_train)
    y_train=encode(y_train).to_numpy()
    y_test=encode(pd.Series(y_test)).to_numpy()    
    y_train=y_train.reshape(y_train.shape[0],1)
    y_test=y_test.reshape(y_test.shape[0],1)


    # define the keras model -Sequential Model "Feed Forward"
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    pd.concat([pd.DataFrame(x_train,columns=x_columns),pd.DataFrame(y_train,columns=y_column)],axis=1).to_csv(path+"preprocessing/train_ann.csv",index=False)
    pd.DataFrame(x_test,columns=x_columns).to_csv(path+"preprocessing/test_ann.csv",index=False)

    np.random.seed(7)
    model = Sequential()
    model.add(Dense(12, input_dim=number_of_dimensions, activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(number_of_dimensions, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(x_train, y_train,validation_data=(x_test, y_test), epochs=200, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(x_train, y_train)
    print('Accuracy: %.2f' % (accuracy*100))

    y_pred=model.predict_classes(x_test)

    confusionmtrix_tocsv(y_test,y_pred,"NeuralNet_",path)
    ##################################################################

    

    
if __name__=='__main__':
    preprocessData("breast-cancer.txt","Results/DataM/Exper5/")