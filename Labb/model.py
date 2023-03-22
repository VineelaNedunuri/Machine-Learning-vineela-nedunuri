import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



def split_data(df, target_col="column", test_size=0.3, random_state=42):
    """Split the dataframe into train, validation and test set

    - train set is used  to fit model
    - validation is used to tune hyperparameter and assess model performance during training
    - test set is used to evaluate the final model performance after training

    """
    X, y = df.drop(target_col, axis=1), df[target_col]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split remaining data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, scale_type='standard'):
    """Scale the dataset with feature by using standardization or normalization """

    if scale_type == 'standard':
        scaler = StandardScaler()
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError('scale must be standard or minmax')
    
    #  fit and transform the training,  transform both validation and test sets
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_val = scaler.transform(X_val)
    scaled_X_test = scaler.transform(X_test)

    return scaled_X_train, scaled_X_val, scaled_X_test


# Grid searchCV  for each model with their accuracy
def grid_search(model, param_grid, X_train, y_train, X_val, y_val, score_file):

    score_file = 'results/accuracy_scores.txt'

    # to create a grid search object and fit that object to training data
    classifier= GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy")
    classifier.fit(X_train, y_train)

    # extract the best model from the grid search object
    model = classifier.best_estimator_

    # predictions on validation data
    y_pred = model.predict(X_val)
   
    # Calculate evaluation score
    accuracy= accuracy_score(y_val, y_pred)
    
    # Check best parameters for each model
    print(f'Best parameters: {classifier.best_params_}')

    # save the accuracy score to a file 
    with open(score_file, 'a') as f:
        f.write(f'{type(model).__name__}: {round(accuracy,2)}\n')

    # print the  classification report for the model
    print(classification_report(y_val, y_pred))
    cm = confusion_matrix(y_val, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Yes", "No"]).plot()
    return cm


