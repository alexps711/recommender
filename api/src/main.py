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
import mxnet as mx
from d2l import mxnet as d2l
from torch.utils.tensorboard import SummaryWriter


class Main:
    def __init__(self, is_svd=False) -> None:
        self.store_features()
        self.is_svd = is_svd
        self.x_train, self.x_test, self.y_train, self.y_test, self.user_indexes, self.event_indexes = self.train_model()
        self.rating_matrix = self.build_utity_matrix()
        self.writer = SummaryWriter()


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
        features = np.array(features) / np.sum(features)
        return features
    
    def store_features(self, max_date=None):
        train['features'] = train.apply(lambda row: self.create_features(row['user_id'], row['event_id']), axis=1)
        print("Stored features")

    def train_model(self):
        user_indexes = []
        event_indexes = []
        y_train = []
        user_indexes = train.user_id
        event_indexes = train.event_id
        x_train, x_test, y_train, y_test = train_test_split(train, train['interested'], test_size=0.2, random_state=42)
        print("Created x_train with size {} and x_test with size {}".format(len(x_train), len(x_test)))
        return x_train, x_test, y_train, y_test, user_indexes, event_indexes
    
    def run(self, user_id=None):
        if self.is_svd:
            print("Using SVD")
            self.x_train = train
            return self.run_svd(user_id)
        else:
            print("Not using SVD")
            return self.run_basic_mf(user_id)
        
    def svd(self, x_train):
        utilMat = np.array(x_train, dtype=float)
        # the nan or unavailable entries are masked
        mask = np.isnan(utilMat)
        masked_arr = np.ma.masked_array(utilMat, mask)
        item_means = np.mean(masked_arr, axis=0)
        # nan entries will replaced by the average rating for each item
        utilMat = masked_arr.filled(item_means)
        x = np.tile(item_means, (utilMat.shape[0],1))
        # The magic happens here. U and V are user and item features
        U, s, V= np.linalg.svd(utilMat, full_matrices=False)
        s=np.diag(s)
        s_root= np.sqrt(s)
        Usk=np.dot(U,s_root)
        skV=np.dot(s_root,V)
        UsV = np.dot(Usk, skV)
        UsV = UsV + x
        return UsV

    def rmse(true, pred):
        # this will be used towards the end
        x = true - pred
        return sum([xi*xi for xi in x])/len(x)

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
                current_features = train[(train.user_id == user) & (train.event_id == event)].features
                # set unavailabe features to 0
                x_train.loc[user, event] = np.nan if current_features.empty else float(current_features.values[0].sum())
        return x_train

    def run_svd(self, user_id=None):
        # if user_id not in train set, we can recommend popular items
        preds = {} #to store the predicted ratings
        self.rating_matrix = self.svd(self.rating_matrix)
        self.rating_matrix = normalize(self.rating_matrix, axis=1)
        if user_id:
            user_index = np.where(self.user_indexes == user_id)[0][0]
            ratings = self.rating_matrix[user_index]
            preds = {int(self.event_indexes[i]): float(ratings[i]) for i in range(len(ratings))}
            sorted_pres = sorted(preds.items(), key=lambda pair: pair[1], reverse=True)
            sorted_pres = dict(sorted_pres)
            return sorted_pres
        else:
            for _, row in self.x_test.iterrows():
                user = row['user_id']
                event = row['event_id']
                user_index = np.where(self.user_indexes == user)[0][0]
                event_index = np.where(self.event_indexes == event)[0][0]
                pred_rating = self.rating_matrix[user_index, event_index]
                # TODO
        return preds
    
    def train_recsys_rating(self, net, num_epochs,
                        devices=d2l.try_all_gpus()):
        loss_fn = torch.nn.MSELoss()
        # init visualizer
        timer = d2l.Timer()
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim
                                =[0, 2],
                                legend=['train loss', 'test RMSE'])
        trainer = torch.optim.SGD(net.parameters(), lr=0.1)
        # format train data
        for epoch in range(5):
            metric, loss_accumulator = d2l.Accumulator(3), 0.0
            train_users = self.x_train['user_id'].values
            train_events = self.x_train['event_id'].values
            train_labels = self.y_train.values
            print("Number of train users: {}".format(len(train_users)))
            # compute predictions
            with autograd.detect_anomaly():
                for user, event in zip(train_users, train_events):
                    timer.start()
                    trainer.zero_grad()
                    print("user: {}, event: {}".format(user, event))
                    user_index = np.where(self.user_indexes == user)[0][0]
                    event_index = np.where(self.event_indexes == event)[0][0]
                    print("user_index: {}, event_index: {}".format(user_index, event_index))
                    user_tensor = Variable(torch.LongTensor([user_index]))
                    event_tensor = Variable(torch.LongTensor([event_index]))
                    pred = net(user_tensor, event_tensor)
                    print("Prediction: {}".format(pred))
                    loss = loss_fn(pred, Variable(torch.FloatTensor([train_labels[user_index]])))
                    print("Loss: {}".format(loss))
                    loss_accumulator += loss.item()
                    loss.backward()
                    trainer.step()
                    timer.stop()
            self.writer.add_scalar("Loss/train", loss_accumulator / len(train_users), epoch)
            print("Total loss for the current epoch: {}".format(loss_accumulator / len(train_users)))
            self.writer.flush()
        self.writer.close()
        #     test_rmse = self.evaluator(net, devices)
        #     train_l = l / 1 # (i + 1)
        #     animator.add(epoch + 1, (loss, test_rmse))
        # print(f'train loss {metric[0] / metric[1]:.3f}, '
        #     f'test RMSE {test_rmse:.3f}')
        # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
        #     f'on {str(devices)}')
    
    def evaluator(self, net, devices=d2l.try_all_gpus()):
        rmse = mx.metric.RMSE()  # Get the RMSE
        rmse_list = []
        test_users = self.x_test['user_id'].values
        test_events = self.x_test['event_id'].values
        test_ratings = self.x_test['interested'].values
        preds = []
        for user, event in zip(test_users, test_events):
            user_index = np.where(self.user_indexes == user)[0][0]
            event_index = np.where(self.event_indexes == event)[0][0]
            user_tensor = Variable(torch.LongTensor([user_index]))
            event_tensor = Variable(torch.LongTensor([event_index]))
            pred = net(user_tensor, event_tensor)
            preds.append(pred)
        rmse.update(labels=test_ratings, preds=preds)
        rmse_list.append(rmse.get()[1])
        return float(np.mean(np.array(rmse_list)))
    
    def run_basic_mf(self, user_id=None):
        n_users = train['user_id'].count()
        n_events = train['event_id'].count()
        net = NeuralCFModel.NeuralCFModel(n_users, n_events, n_factors=5)
        lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
        self.train_recsys_rating(net, num_epochs)
        user_index = np.where(self.user_indexes == user_id)[0][0]
        user_tensor = Variable(torch.LongTensor([user_index]))
        ratings = []
        for i, row in train.iterrows():
            event_index = np.where(self.event_indexes == row['event_id'])[0][0]
            event_tensor = Variable(torch.LongTensor([event_index]))
            rating = net(user_tensor, event_tensor)
            ratings.append(rating.item())
        train['rating'] = ratings
        # normalize data
        train['rating'] = train['rating'].values / np.max(train['rating'].values)
        evs = train[['event_id', 'rating']].sort_values(by='rating', ascending=False)
        return dict(evs.values)
        # if user_id not in train set, we can recommend popular items
        # n_users = train['user_id'].nunique()
        # n_events = train['event_id'].nunique()
        # model = NeuralCFModel.NeuralCFModel(n_users, n_events, n_factors=5)
        # loss_fn = torch.nn.MSELoss() 
        # optimizer = torch.optim.SGD(model.parameters(),
        #                          lr=1e-6)
        # for _, row in self.x_train.iterrows():
        #     rating = Variable(torch.FloatTensor([row['features'].sum()]))
        #     user_index = np.where(self.user_indexes == row.user_id)[0][0]
        #     event_index = np.where(self.event_indexes == row.event_id)[0][0]
        #     user = Variable(torch.LongTensor([user_index]))
        #     event = Variable(torch.LongTensor([event_index]))
        #     prediction = model.predict(user, event)
        #     print(row.user_id, row.event_id, prediction)
        #     print(rating)
        #     loss = loss_fn(prediction, rating)
        #     loss.backward()
        #     optimizer.step()
            
        # for _, row in self.x_test.iterrows():
        #     user_index = np.where(self.user_indexes == row.user_id)[0][0]
        #     event_index = np.where(self.event_indexes == row.event_id)[0][0]
        #     user = Variable(torch.LongTensor([user_index]))
        #     event = Variable(torch.LongTensor([event_index]))
        #     prediction = model.predict(user, event)
        
    # def run():
        # x_train, x_test, y_train, y_test, user_indexes, event_indexes = get_train_data()  
        
        # regular MF approach
        # n_users = train['user_id'].nunique()
        # n_events = train['event_id'].nunique()
        # model = NeuralCFModel.NeuralCFModel(n_users, n_events, n_factors=5)
        # loss_fn = torch.nn.MSELoss() 
        # optimizer = torch.optim.SGD(model.parameters(),
        #                         lr=1e-6)
        # for _, row in x_train.iterrows():
        #     rating = Variable(torch.FloatTensor([row['features'].sum()]))
        #     user_index = np.where(user_indexes == row.user_id)[0][0]
        #     event_index = np.where(event_indexes == row.event_id)[0][0]
        #     user = Variable(torch.LongTensor([user_index]))
        #     event = Variable(torch.LongTensor([event_index]))
        #     prediction = model.predict(user, event)
        #     loss = loss_fn(prediction, rating)
        #     loss.backward()
        #     optimizer.step()
        
        # for _, row in x_test.iterrows():
        #     user_index = np.where(user_indexes == row.user_id)[0][0]
        #     event_index = np.where(event_indexes == row.event_id)[0][0]
        #     user = Variable(torch.LongTensor([user_index]))
        #     event = Variable(torch.LongTensor([event_index]))
        #     prediction = model.predict(user, event)
        #     print(prediction)
        #     print(rating)
            
        # SVD approach
        # x_train = build_utity_matrix()
        # x_train = normalize(x_train, 'l1', axis=1)
        # svdout = svd(x_train)
        # pred = [] #to store the predicted ratings
        # for _, row in x_test.iterrows():
        #     user = row['user_id']
        #     item = row['event_id']
        #     user_index = np.where(user_indexes == user)[0][0]
        #     event_index = np.where(event_indexes == item)[0][0]
        #     pred_rating = svdout[user_index, event_index]
        #     pred.append(pred_rating)
        # print(pred)    
        # print(rmse(y_test, pred))