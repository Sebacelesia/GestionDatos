from sklearn.model_selection import train_test_split

def preprocess(dataset):
    X = dataset.drop(columns=["churn","registration_date","last_seen"])
    y = dataset["churn"]
    return X, y

def hacer_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Devuelve X_train, X_test, y_train, y_test.
    """
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
    return X_train, X_test, y_train, y_test
