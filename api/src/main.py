from models import NeuralCFModel
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.autograd import Variable
import torch
from torch import autograd
from sklearn.preprocessing import normalize
from data_parser import train
import api
from torch.utils.tensorboard import SummaryWriter
from models import FunkSVDModel


class Main:
    def __init__(self) -> None:
        # log model loss
        self.feature_matrix = self.store_features()
        self.writer = SummaryWriter()
        self.x_train, self.x_test, self.y_train, self.y_test, self.user_indexes, self.event_indexes = self.get_model_data()
        self.init_basic_mf()
        self.init_svd()
        
        
    def init_basic_mf(self):
        self.n_users = train['user_id'].nunique()
        self.n_events = train['event_id'].nunique()
        self.basic_model = NeuralCFModel.NeuralCFModel(
            self.n_users, self.n_events, n_factors=5)
        num_epochs = 20
        self.train_basic_mf_model(num_epochs)
        
    def init_svd(self):
        self.rating_matrix = self.build_utity_matrix()
        self.svd_model = FunkSVDModel.FunkSVDModel()
        self.svd_model.fit(self.rating_matrix)
        
    def create_features(self, user_id, event_id):
        print("Creating featues for user:", user_id, "and event:", event_id)
        # 0 - users attending event
        # 1 - friends attending event
        # 2 - friends attending event / number of friends
        # 3 - location
        # 4 - age
        features = [0, 0, 0, 0, 0]
        # 0 - user attending events
        event_attendees = api.get_event_attendees(event_id)
        features[0] = len(event_attendees)
        # 1 - friends attending events
        user_friends = api.get_user_friends(user_id)
        friends_attending_event = np.intersect1d(user_friends, event_attendees)
        features[1] = len(friends_attending_event)
        # 2 - friends attending event / number of friends
        if user_friends:
            features[2] = len(friends_attending_event) / len(user_friends)
        else:
            features[2] = 0
        # 3 - location
        features[3] = api.get_location_similarity(user_id, event_id)
        # 4 - age
        features[4] = api.get_age_similarity(user_id, event_attendees)
        return features

    def store_features(self, max_date=None):
        feature_matrix = []
        for i in range(train.shape[0]):
            user_id = train.iloc[i].user_id
            event_id = train.iloc[i].event_id
            feature = self.create_features(user_id, event_id)
            feature_matrix.append(feature)
        feature_matrix = normalize(feature_matrix)
        print("Normalized and stored features")
        return feature_matrix

    def get_model_data(self):
        user_indexes = []
        event_indexes = []
        y_train = []
        user_indexes = train.user_id.unique()
        event_indexes = train.event_id.unique()
        x_train, x_test, y_train, y_test = train_test_split(
            train, train['interested'], test_size=0.2, random_state=42)
        print("Created x_train with size {} and x_test with size {}".format(
            len(x_train), len(x_test)))
        return x_train, x_test, y_train, y_test, user_indexes, event_indexes

    def build_utity_matrix(self):
        # construct utility matrix
        print("Building utility matrix...")
        unique_users = train.user_id.unique()
        unique_events = train.event_id.unique()
        x_train = pd.DataFrame(index=unique_users, columns=unique_events)
        # fill matrix with ratings
        for user in unique_users:
            event_attending = train[train.user_id == user].event_id.unique()
            for event in event_attending:
                current_features_index = train[(train.user_id == user) & (
                    train.event_id == event)].index[0]
                current_features = self.feature_matrix[current_features_index]
                # set unavailabe features to 0
                x_train.loc[user, event] = np.nan if len(
                    current_features) < 1 else float(np.sum(current_features))
        return x_train
    
    def train_basic_mf_model(self, num_epochs):
        print("Training model...")
        loss_fn = torch.nn.MSELoss()
        # init visualizer
        trainer = torch.optim.SGD(self.basic_model.parameters(), lr=0.001)
        # format train data
        for epoch in range(num_epochs):
            loss_accumulator = 0.0
            train_users = self.x_train['user_id'].values
            train_events = self.x_train['event_id'].values
            train_labels = self.y_train.values
            # compute predictions
            with autograd.detect_anomaly():
                for user, event in zip(train_users, train_events):
                    trainer.zero_grad()
                    user_index = np.where(self.user_indexes == user)[0][0]
                    event_index = np.where(self.event_indexes == event)[0][0]
                    user_tensor = Variable(torch.LongTensor([user_index]))
                    event_tensor = Variable(torch.LongTensor([event_index]))
                    pred = self.basic_model(user_tensor, event_tensor)
                    loss = loss_fn(pred, Variable(
                        torch.FloatTensor([train_labels[user_index]])))
                    loss_accumulator += loss.item()
                    loss.backward()
                    trainer.step()
            self.writer.add_scalar(
                "Loss/train", loss_accumulator / len(train_users), epoch)
            print("Total loss for the current epoch: {}".format(
                loss_accumulator / len(train_users)))
            self.writer.flush()
        self.writer.close()
        print("Finished training model.")

    def run_svd(self, user_id=None):
        user_index = np.where(self.user_indexes == user_id)[0][0]
        ratings = self.svd_model.predict_instance(user_index)
        evs = {self.event_indexes[i]: rating for i,
               rating in enumerate(ratings)}
        return evs

    def run_basic_mf(self, user_id=None):
        user_index = np.where(self.user_indexes == user_id)[0][0]
        user_tensor = Variable(torch.LongTensor([user_index]))
        ratings = []
        for i, row in train.iterrows():
            event_index = np.where(self.event_indexes == row['event_id'])[0][0]
            event_tensor = Variable(torch.LongTensor([event_index]))
            rating = self.basic_model(user_tensor, event_tensor)
            ratings.append(rating.item())
        train['rating'] = ratings
        # normalize data
        train['rating'] = train['rating'].values / \
            np.max(train['rating'].values)
        evs = train[['event_id', 'rating']].sort_values(
            by='rating', ascending=False)
        return dict(evs.values)
    
    def run(self, user_id=None, is_svd=False):
        if is_svd:
            print("Using SVD")
            # train FunkSVD model
            return self.run_svd(user_id)
        else:
            print("Not using SVD")
            return self.run_basic_mf(user_id)
