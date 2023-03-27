import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  accuracy_score,f1_score
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
    pipe_logistic = Pipeline([("scaler", scaler), ("LR", LogisticRegression(random_state=42))])
    pipe_KNN = Pipeline([("scaler", scaler), ("KNN", KNeighborsClassifier())])
    pipe_tree= Pipeline([("scaling", scaler), ("DT", DecisionTreeClassifier(random_state=42))])
    pipe_forest = Pipeline([("scaling", scaler), ("RF", RandomForestClassifier(random_state=42))])
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
def grid_search(pipeline,param_grid, X_train, y_train, X_val, y_val,dataset_name='dataset1',score_file='results/accuracy_scores.txt'):

    grid_search= GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="f1",cv=3, verbose=1,error_score='raise')

    # to fit that object to training data
    grid_search.fit(X_train, y_train)


    # predictions on validation data
    y_pred =grid_search.predict(X_val)

    # Calculate evaluation score
    accuracy= accuracy_score(y_val, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")

   # calculate evaluation score
    f1 = f1_score(y_val, y_pred)
    print(f"F1 Score: {f1:.2f}")

    # Check best parameters for each model
    print(f'Best parameters: {grid_search.best_params_}')

    # save the accuracy score to a file 
    with open(score_file, 'a') as f:
        f.write(f'{dataset_name}: F1 score - {f1:.2f}, accuracy-{accuracy * 100:.2f}% \n')

    return y_pred



 

# To calculate the classification report and confusion matrix
def evaluate_classification(model_name,y_val, y_pred):
    print(f'CLASSIFICATION REPORT:\n{classification_report(y_val, y_pred)}')
    print("................................\n")
    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix: \n ",cm)
    print("................................\n")
    matrix= ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
    matrix.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    return cm

