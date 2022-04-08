### Recommender system ###

A matrix-factorization based approach to recommending events.

This project contains both an API that train the models and returns recommendations, and a web application to interact with the recommendations.
To start up the API, run the following commands in order: 
```
cd api & pip3 install requirements.txt
export FLAS_APP = src/app
flask run
```
To start the web application, run the following commands in order:
```
cd app & npm install
npm start
```

The dataset is available to download here: https://www.kaggle.com/competitions/event-recommendation-engine-challenge/data
