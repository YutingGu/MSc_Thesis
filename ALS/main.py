# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-21 17:32:24
@Last Modified by:   tushushu
@Last Modified time: 2018-08-21 17:32:24
"""
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
import pandas as pd
import numpy as np
from recommend.als import ALS
from utils.utils import run_time


def format_prediction(item_id, score):
    return "item_id:%d score:%.2f" % (item_id, score)


@run_time
def main():
    print("Tesing the performance of ALS...")
    # Load data
    data_train_load = pd.read_csv('../splitdata/data_train.csv')
    #X = [[row[col] for col in data_train_load.columns[:3]] for row in data_train_load.to_dict('records')]
    X = data_train_load.to_numpy().tolist()
    # Train model
    model = ALS()
    model.fit(X, k=3, max_iter=10)
    print()

    print("Showing the predictions of users...")
    # Predictions
    user_ids = range(1, 5)
    predictions = model.predict(user_ids, n_items=2)
    for user_id, prediction in zip(user_ids, predictions):
        _prediction = [format_prediction(item_id, score)
                       for item_id, score in prediction]
        print("User id:%d recommedation: %s" % (user_id, _prediction))


if __name__ == "__main__":
    main()
