from curses import noraw
import pandas as pd
import numpy as np
from helpers import memoized
import random

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