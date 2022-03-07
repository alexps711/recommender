import pandas as pd
import numpy as np
from helpers import memoized

@memoized
def get_users():
    users = pd.read_csv('data/users.csv')
    return users

@memoized
def get_events():
    events = pd.read_csv('data/events.csv.gz', compression="gzip", index_col=0)
    return events

@memoized
def get_event_attendees():
    event_attendees = pd.read_csv('data/event_attendees.csv.gz', compression="gzip", index_col=0)
    return event_attendees

@memoized
def get_user_friends():
    user_friends = pd.read_csv('data(/user_friends.csv.gz', compression="gzip", index_col=0)
    return user_friends

@memoized
def get_train_data():
    train_data = pd.read_csv('data/train.csv')
    return train_data

@memoized
def get_test_data():
    test_data = pd.read_csv('data/test.csv')
    return test_data