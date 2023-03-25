import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def split_data(df, target_col="column", test_size=0.2, random_state=42):
    """Split the dataframe into train, validation and test set

    - train set is used  to fit model
    - validation is used to tune hyperparameter and assess model performance during training
    - test set is used to evaluate the final model performance after training

    """
    X, y = df.drop(target_col, axis=1), df[target_col]

    # Split data into train and test sets (80/20 split)
    X_train, X_t, y_train, y_t = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split remaining data 20%  into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_t, y_t, test_size=0.2, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(scale_type='standard'):
    """Scale the dataset with feature by using standardization or normalization """

    if scale_type == 'standard':
        scaler = StandardScaler()
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError('scale must be standard or minmax')
 # Create pipelines
    pipe_logistic = Pipeline([("scaler", scaler), ("LR", LogisticRegression())])
    pipe_KNN = Pipeline([("scaler", scaler), ("KNN", KNeighborsClassifier())])
    pipe_tree= Pipeline([("scaling", scaler), ("DT", DecisionTreeClassifier())])
    pipe_forest = Pipeline([("scaling", scaler), ("RF", RandomForestClassifier())])
    pipe_Gaussian = Pipeline([("scaling", scaler), ("NB", GaussianNB())])

    # Return pipelines as a dictionary
    pipelines = {
        'Logistic Regression': pipe_logistic,
        'K-Nearest Neighbor': pipe_KNN,
        'Decision Tree': pipe_tree,
        'Random Forest': pipe_forest,
        'Gaussian Naive Bayes': pipe_Gaussian
    }

    return pipelines


# Grid searchCV  for each model with their accuracy
def grid_search(pipeline,param_grid, X_train, y_train, X_val, y_val, score_file):

    grid_search= GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="accuracy", cv=3,error_score='raise')

    # to fit that object to training data
    grid_search.fit(X_train, y_train)

    # extract the best model from the grid search object
    model = grid_search.best_estimator_

    # predictions on validation data
    y_pred = model.predict(X_val)
   
    # Calculate evaluation score
    accuracy= accuracy_score(y_val, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")
    
    # Check best parameters for each model
    print(f'Best parameters: {grid_search.best_params_}')

    # save the accuracy score to a file 
    with open(score_file, 'a') as f:
        f.write(f'model: {accuracy * 100:.2f}%\n')
    
    
    
    
    
    #print classification report
    #print(f'CLASSIFICATION REPORT:\n{classification_report(y_val, y_pred)}')
    #cm = confusion_matrix(y_val, y_pred)
    #print(f"Confusion Matrix: \n ",cm)
    #ConfusionMatrixDisplay(cm, display_labels=["Yes", "No"]).plot()


