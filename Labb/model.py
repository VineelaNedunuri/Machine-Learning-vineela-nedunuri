import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def train_val_test_split(df, target_col="column", test_size=0.3, random_state=42):
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
def grid_search(model,param_grid,X_train, y_train, X_val, y_val):

    classifier= GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy")
    classifier.fit(X_train, y_train)

    # predictions on validation data
    y_pred = classifier.predict(X_val)

    # Calculate evaluation score
    accuracy= accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(f" accuracy:{accuracy:.2f}\nprecision:{precision:.2f}\nrecall:{recall:.2f}\nf1_score:{f1:.2f}")

    # Check best parameters for each model
    print(f'Best parameters: {classifier.best_params_}')




    
    
