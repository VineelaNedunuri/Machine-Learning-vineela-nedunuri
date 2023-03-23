import pandas as pd
from sklearn.model_selection import train_test_split
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


# Grid searchCV  for each model with their accuracy
def grid_search(pipe_model, param_grid, X_train, y_train, X_val, y_val, score_file):


    # to create a grid search object and fit that object to training data
    classifier= GridSearchCV(estimator=pipe_model, param_grid=param_grid, cv=5, scoring="accuracy")
    classifier.fit(X_train, y_train)

    # extract the best model from the grid search object
    model = classifier.best_estimator_

    # predictions on validation data
    y_pred = model.predict(X_val)
   
    # Calculate evaluation score
    accuracy= accuracy_score(y_val, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")
    
    # Check best parameters for each model
    print(f'Best parameters: {classifier.best_params_}')

    # save the accuracy score to a file 
    with open(score_file, 'a') as f:
        f.write(f'{type(model).__name__}: {accuracy * 100:.2f}\n')
    
    #print classification report
    print(f'CLASSIFICATION REPORT:\n{classification_report(y_val, y_pred)}')
    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix: \n ",cm)
    ConfusionMatrixDisplay(cm, display_labels=["Yes", "No"]).plot()


