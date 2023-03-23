import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_col="column", test_size=0.2, random_state=42):
    """Split the dataframe into train, validation and test set

    - train set is used  to fit model
    - validation is used to tune hyperparameter and assess model performance during training
    - test set is used to evaluate the final model performance after training

    """


    X, y = df.drop(target_col, axis=1), df[target_col]

    # Split data into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split remaining data 20%  into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

