import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train_val_test_split(df, target_col="column", test_size=0.2, random_state=42):
    """Split the dataframe into train, validation and test set

    - train set is used  to fir model
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
        X_test, y_test, test_size=test_size, random_state=random_state
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





    
    
