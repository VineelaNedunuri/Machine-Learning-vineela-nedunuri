import pandas as pd 
from sklearn.model_selection import train_test_split


def train_val_test_split(df, column, test_size=0.2, random_state=42):
    """Split the dataframe into train, validation and test set

    - train set is used  to fir model
    - validation is used to tune hyperparameter and assess model performance during training
    - test set is used to evaluate the final model performance after training

    """
    X, y = df.drop("column", axis=1), df["column"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split remaining data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
