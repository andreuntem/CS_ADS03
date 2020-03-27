import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def preprocess(df):
    """This function takes a dataframe and preprocesses it so it is
    ready for the training stage.

    The DataFrame contains columns used for training (features)
    as well as the target column.

    It also contains some rows for which the target column is unknown.
    Those are the observations you will need to predict for KATE
    to evaluate the performance of your model.

    Here you will need to return the training set: X and y together
    with the preprocessed evaluation set: X_eval.

    Make sure you return X_eval separately! It needs to contain
    all the rows for evaluation -- they are marked with the column
    evaluation_set. You can easily select them with pandas:

         - df.loc[df.evaluation_set]

    For y you can either return a pd.DataFrame with one column or pd.Series.

    :param df: the dataset
    :type df: pd.DataFrame
    :return: X, y, X_eval
    """

    # Select all possible columns
    selected_columns = ['state', 'evaluation_set', 'goal', 'static_usd_rate',
                        'country', 'category', 'deadline', 'launched_at']
    df = df[selected_columns]

    # Select subset of variables
    subset = ['goal_usd', 'country', 'category_name', 'diff_days']

    if 'goal_usd' in subset:
        df['goal_usd'] = df['goal']*df['static_usd_rate']
        df['goal_usd'] = df['goal'].apply(np.log)
    df.drop(['goal', 'static_usd_rate'], axis=1, inplace=True)

    # >>> Country
    if 'country' not in subset:
        df.drop('country', axis=1, inplace=True)

    # >>> Category
    if 'category_color' in subset:
        df['category_color'] = df['category'].apply(lambda x: json.loads(x)['color'])
    if 'category_name' in subset:
        df['category_name'] = df['category'].apply(lambda x: json.loads(x)['name'])
    if 'category_position' in subset:
        df['category_position'] = df['category'].apply(lambda x: json.loads(x)['position'])
    df.drop(['category'], axis=1, inplace=True)

    # >>> Days
    if 'diff_days' in subset:
        df['deadline'] = df['deadline'].apply(datetime.utcfromtimestamp)
        df['launched_at'] = df['launched_at'].apply(datetime.utcfromtimestamp)
        df['diff_days'] = (df['deadline']-df['launched_at']).apply(lambda x: x.days)
    df.drop(['deadline', 'launched_at'], axis=1, inplace=True)

    # Generate dummies and create X, X_eval and y
    df = pd.get_dummies(df)
    X = df[~df['evaluation_set']].drop(columns=['state', 'evaluation_set'], axis=1)
    X_eval = df[df['evaluation_set']].drop(columns=['state', 'evaluation_set'], axis=1)
    y = df[~df['evaluation_set']]['state']

    # Standardize X and X_eval
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_eval = scaler.transform(X_eval)

    return X, y, X_eval


def train(X, y):
    """Trains a new model on X and y and returns it.

    :param X: your processed training data
    :type X: pd.DataFrame
    :param y: your processed label y
    :type y: pd.DataFrame with one column or pd.Series
    :return: a trained model
    """
    model = LogisticRegression(solver='liblinear', n_jobs=-1)
    model.fit(X, y)
    return model


def predict(model, X_test):
    """This functions takes your trained model as well
    as a processed test dataset and returns predictions.

    On KATE, the processed test dataset will be the X_eval you built
    in the "preprocess" function. If you're testing your functions locally,
    you can try to generate predictions using a sample test set of your
    choice.

    This should return your predictions either as a pd.DataFrame with one column
    or a pd.Series

    :param model: your trained model
    :param X_test: a processed test set (on KATE it will be X_eval)
    :return: y_pred, your predictions
    """
    y_pred = model.predict(X_test)
    return y_pred
