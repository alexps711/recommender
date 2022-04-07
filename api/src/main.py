from cProfile import label
import os
from models import NeuralCFModel
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.autograd import Variable
import torch
from torch import autograd
from sklearn.preprocessing import normalize
from data_parser import train, events
import api
from torch.utils.tensorboard import SummaryWriter
from models import FunkSVDModel
import recmetrics
import ml_metrics as metrics
from collections import Counter
from sklearn.metrics import fbeta_score
import plotly.express as px


class Main:
    def __init__(self) -> None:
        # create and save feature vectors
        self.feature_matrix = self.store_features()
        # init a TensorBoard writer object
        self.writer = SummaryWriter()
        # load train and test data
        self.x_train, self.x_test, self.y_train, self.y_test, self.user_indexes, self.event_indexes = self.get_model_data()
        # init models
        self.init_basic_mf()
        self.init_svd()
        self.mapk()
        self.novelty_test()
     
    def per_test(self):
        """Personalization test.
        """
        evs = list(self.run_neural_mf(user_id=3044012).keys())[:100]
        evs2 = list(map(lambda x: x[0], sorted([(k,v) for k,v in self.run_svd(user_id=3044012).items()], key=lambda x: x[1], reverse=True)[:100]))
        print([evs, evs2])
        print(recmetrics.personalization([evs, evs2]))
        
    def intra_test(self):
        """"Intra-list test."""
        evs = list(self.run_neural_mf(user_id=3044012).keys())[:100]
        evs2 = list(self.run_neural_mf(user_id=23150045).keys())[:100]
        evs3 = list(self.run_neural_mf(user_id=24978365).keys())[:100]
        all_evs = [] + evs + evs2 + evs3
        df = events[events['event_id'].isin(all_evs)]
        df.set_index('event_id', inplace=True)
        df.fillna(0, inplace=True)
        df.drop('start_time', axis=1, inplace=True)
        df = pd.get_dummies(df, prefix=['city', 'state', 'zip', 'country'])
        print(recmetrics.intra_list_similarity([evs, evs2, evs3], df))
        
    def novelty_test(self):
        evs = list(self.run_neural_mf(user_id=3044012).keys())[:100]
        evs2 = list(self.run_neural_mf(user_id=23150045).keys())[:100]
        occurences = train['event_id'].value_counts().to_dict()
        print(recmetrics.novelty([evs, evs2], occurences, u=train['user_id'].nunique(), n=100))
        
        
    def mapk(self):
        """ Compute the MAP@k for the neural model."""
        neural_scores = []
        svd_scores = []
        mar_neural_scores = []
        mar_svd_scores = []
        fb_neural_scores = []
        fb_svd_scores = []
        
        for i in range(20):
            uid1 = train.sample().user_id.values[0]
            uid2 = train.sample().user_id.values[0]
            
            evs = list(self.run_neural_mf(user_id=uid1).keys())[:20]
            evs2 = list(self.run_neural_mf(user_id=uid2).keys())[:20]
            Y_evs = train[train['event_id'].isin(evs) & train['interested'] == 1]['event_id'].values.tolist()[:20]
            Y_evs2 = train[train['event_id'].isin(evs2) & train['interested'] == 1]['event_id'].values.tolist()[:20]
            score = metrics.mapk([Y_evs, Y_evs2], [evs, evs2], k=10)
            mar_score  = recmetrics.mark([Y_evs, Y_evs2], [evs, evs2], k=10)
            fb = fbeta_score(sorted(Y_evs), sorted(evs[:len(Y_evs)]), beta=1, average='weighted')
            print("Neural MAP@20:", score, "MAR@20:", mar_score, "Fbeta:", fb)
            neural_scores.append(score)
            mar_neural_scores.append(mar_score)
            fb_neural_scores.append(fb)    
            
            evs = list(map(lambda x: x[0], list(Counter(self.run_svd(user_id=uid1)).most_common())))[:20]
            evs2 = list(map(lambda x: x[0], list(Counter(self.run_svd(user_id=uid2)).most_common())))[:20]
            Y_evs = train[train['event_id'].isin(evs) & train['interested'] == 1]['event_id'].values.tolist()[:20]
            Y_evs2 = train[train['event_id'].isin(evs2) & train['interested'] == 1]['event_id'].values.tolist()[:20]
            score = metrics.mapk(sorted([Y_evs, Y_evs2]), [evs, evs2], k=10)
            mar_score  = recmetrics.mark([Y_evs, Y_evs2], [evs, evs2], k=10)
            fb = fbeta_score(sorted(Y_evs), sorted(evs[:len(Y_evs)]), beta=1, average='weighted')
            print("SVD MAP@20:", score, "MAR@20:", mar_score, "Fbeta:", fb)
            svd_scores.append(score)
            mar_svd_scores.append(mar_score)
            fb_svd_scores.append(fb)
            
        print("Neural MAP@20:", np.mean(neural_scores), "MAR@20:", np.mean(mar_neural_scores), "Fbeta:", np.mean(fb_neural_scores))
        df = pd.DataFrame({'MAP@K - Neural': neural_scores, 'MAP@K - SVD': svd_scores, 'MAR@K - Neural': mar_neural_scores, 'MAR@K - SVD': mar_svd_scores, 'Fbeta - Neural': fb_neural_scores, 'Fbeta - SVD': fb_svd_scores})
        px.line(df).show()
        
    def init_basic_mf(self):
        """ Initialize and train a neural collaborative filtering model """
        self.n_users = train['user_id'].nunique()
        self.n_events = train['event_id'].nunique()
        self.basic_model = NeuralCFModel.NeuralCFModel(
            self.n_users, self.n_events, n_factors=5)
        if os.path.exists('models/basic_model.pth'):
            self.basic_model.load_state_dict(torch.load('models/basic_model.pth'))
        else:
            num_epochs = 50
            self.train_neural_model(num_epochs)
            torch.save(self.basic_model.state_dict(), 'models/basic_model.pth')
        
    def init_svd(self):
        """Initialize and train a FunkSVD model"""
        self.rating_matrix = self.build_utity_matrix()
        self.svd_model = FunkSVDModel.FunkSVDModel()
        self.svd_model.train(self.rating_matrix)
        
    def create_features(self, user_id, event_id):
        """Create a feature vector for a given user and event.
        The feature vector looks like:
        0 - users attending event
        1 - friends attending event
        2 - friends attending event / number of friends
        3 - location
        4 - age
        
        Args:
            user_id (int): The id of the user.
            event_id (int): The id of the event.

        Returns:
            int[][][]: A feature matrix with the feature vector at the [user_id, event_id] position.
        """
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

    def store_features(self):
        """Create and save feature vectors for all users and events.
        
        Returns:
            int[][]: A 2d array with the feature vectors for all users and events.
        """
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
        """Split the data between train and test.

        Returns:
            any: a collection of train and test data, plus the indices (location) of the users and events.
        """
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
        """Build a rating matrix from the train data.
        
        Returns:
            int[][]: An feedback matrix with scalar ratings for all users and events.
        """
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
    
    def train_neural_model(self, num_epochs):
        """Train the neural collaborative filtering model.

        Args:
            num_epochs (int): The number of iterations to go through.
        """
        print("Training model...")
        loss_fn = torch.nn.MSELoss()
        # init visualizer
        trainer = torch.optim.SGD(self.basic_model.parameters(), lr=0.008)
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

    def run_svd(self, user_id):
        """Compute the rating of events for a given user.

        Args:
            user_id (int): The id of the user.

        Returns:
            dict: A dictionary of events with their ratings.
        """
        user_index = np.where(self.user_indexes == user_id)[0][0]
        ratings = self.svd_model.predict(user_index)
        evs = {self.event_indexes[i]: rating for i,
               rating in enumerate(ratings)}
        return evs

    def run_neural_mf(self, user_id):
        """Compute the rating of events for a given user.

        Args:
            user_id (int): The id of the user.

        Returns:
            dict: A dictionary of events with their ratings.
        """
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
    
    def run(self, user_id, is_svd=False):
        """Predict ratings for a given user.

        Args:
            user_id (int): The id of the user.
            is_svd (bool, optional): If True, FunkSVD model will be used, neural 
            collaborative filtering otherwise. Defaults to False.

        Returns:
            dict: A dictionary of events with their ratings.
        """
        if is_svd:
            print("Using SVD")
            # train FunkSVD model
            return self.run_svd(user_id)
        else:
            print("Not using SVD")
            return self.run_neural_mf(user_id)
