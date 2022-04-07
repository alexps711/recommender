from curses import noraw
import pandas as pd
import numpy as np
from helpers import memoized
import random
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import recmetrics


p = 1  # 1% of the lines

@memoized
def get_users():
    print("Reading users...")
    users = pd.read_csv('../data/users.csv')
    return users

@memoized
def get_events():
    print("Reading events...")
    events = pd.read_csv('../data/events.csv.gz', compression="gzip")
    return events

@memoized
def get_event_attendees():
    print("Reading event_attendees...")
    event_attendees = pd.read_csv('../data/event_attendees.csv.gz')
    event_attendees.rename(columns = {'event':'event_id'}, inplace = True)
    return event_attendees

@memoized
def get_user_friends():
    print("Reading user_friends...")
    user_friends = pd.read_csv('../data/user_friends.csv.gz', compression="gzip")
    user_friends.rename(columns = {'user':'user_id'}, inplace = True)
    return user_friends

@memoized
def get_train_data(): 
    print("Reading train data...")
    train_data = pd.read_csv('../data/train.csv', nrows=2000)
    train_data.rename(columns = {'user':'user_id'}, inplace = True)
    train_data.rename(columns = {'event':'event_id'}, inplace = True)
    return train_data

@memoized
def get_test_data():
    print("Reading test data...")
    test_data = pd.read_csv('../data/test.csv')
    test_data.rename(columns = {'user':'user_id'}, inplace = True)
    test_data.rename(columns = {'event':'event_id'}, inplace = True)
    return test_data


# load data
users = get_users()
events = get_events()
attendees = get_event_attendees()
friends = get_user_friends()
train = get_train_data()
test = get_test_data()

# if __name__ == '__main__':
    # df = events
    # categories = ['bar', 'club', 'restaurant', 'festival', 'other']
    # X = df[[f"c_{i}" for i in range(1, 100)]]

    # pca = PCA(n_components=3)
    # components = pca.fit_transform(X)

    # total_var = pca.explained_variance_ratio_.sum() * 100

    # fig = px.scatter_3d(
    #     components, x=df['c_1'], y=df['c_2'], z=df['c_3'],
    #     title=f'Total Explained Variance: {total_var:.2f}%',
    # )
    # fig.show()
    # long tail plot
    # df = pd.concat([train['user_id'], train['event_id']], axis=1)
    # print(df)
    # recmetrics.long_tail_plot(df, item_id_column='event_id', interaction_type='interactions', percentage=0.1)