# import wasabi
# from constants import USER
# from os import error
# from nltk.corpus.reader.ppattach import PPAttachment
# import config
# import json
import json
from flask import Flask, request, jsonify
import pandas as pd
# from context_awareness import explicit, history, collective, season, transport
# from helpers import format_records
# from nlp import parser
# from fetch import search_by_tokens
from main import Main
from flask_cors import CORS, cross_origin
from data_parser import users, train, events
import api
from mxnet import npx
import matplotlib

def create_app():
    app = Flask(__name__)
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    model = Main()
    matplotlib.use('Agg')
    npx.set_np()
    
    @app.route('/users', methods=['GET'])
    @cross_origin()
    def get_users():
        user_id = request.args.get('id')
        train_filter = request.args.get('train') == 'true'
        if train_filter:
            filtered_users = train[train['user_id'].apply(str).str.startswith(user_id)]
        else:
            filtered_users = users[users['user_id'].apply(str).str.startswith(user_id)]
        return json.dumps(filtered_users['user_id'].unique()[:100].tolist())
    
    @app.route('/user', methods=['GET'])
    @cross_origin()
    def get_user():
        user_id = request.args.get('id')
        user = api.get_user(int(user_id))
        return user.to_json(orient='records')
    
    @app.route('/events', methods=['GET'])
    @cross_origin()
    def get_events():
        user_id = request.args.get('id')
        is_svd = request.args.get('svd') == 'true'
        event_rating_dict = model.run(int(user_id), is_svd)
        evs = events[events['event_id'].isin(event_rating_dict.keys())]
        evs['rating'] = evs['event_id'].apply(lambda x: event_rating_dict[x])
        return evs.to_json(orient='records')
            
    @app.route('/prevevents', methods=['GET'])
    @cross_origin()
    def get_prev_events():
        user_id = request.args.get('id')
        prev_events = api.get_user_events(int(user_id))
        evs = events[events['event_id'].isin(prev_events)]
        return evs.to_json(orient='records')
            
    
    # @app.route('/', methods=['POST'])
    # def main():
    #     try:
    #         uid: int = request.form['uid']
    #     except:
    #         return "User id (uid) and location (location) parameters not supplied."
    #     try:
    #         lat = request.form['lat']
    #     except:
    #         return "Latitude (lat) parameter not supplied."
    #     try:
    #         lon = request.form['lon']
    #     except:
    #         return "Longitude (lon) parameter not supplied."
    #     location = {'lat': float(lat), 'lon': float(lon)}
    #     records = explicit.rank_explicit(uid, location)
    #     records_2 = history.rank_history(uid, records)
    #     records_3 = collective.rank_collective(uid, records_2, location)
    #     records_4 = season.rank_seasonal(uid, records_3, location)
    #     records_5 = transport.rank_transport(uid, records_4, location)
    #     to_return = format_records(records_5)

    #     return jsonify(to_return)

    # @app.route('/search', methods=['POST'])
    # def search():
    #     try:
    #         uid: int = request.form['uid']
    #         USER.ID = uid
    #     except:
    #         return "User id (uid) and location (location) parameters not supplied."
    #     try:
    #         lat = request.form['lat']
    #     except:
    #         return "Latitude (lat) parameter not supplied."
    #     try:
    #         lon = request.form['lon']
    #     except:
    #         return "Longitude (lon) parameter not supplied."
    #     try:
    #         sentence: str = request.form['sentence']
    #     except:
    #         return "Sentence (sentence) parameter not supplied"
    #     try:
    #         filters: dict = json.loads(request.form['filters'])
    #     except:
    #         filters = None

    #     location = {'lat': float(lat), 'lon': float(lon)}
    #     tokens = parser.parse(sentence)
    #     washed_tokens = parser.wash(tokens)
    #     records = search_by_tokens(washed_tokens, location, filters)
    #     if(len(records.items()) > 0):
    #         for category, recs in records.items():
    #             for rec in recs:
    #                 index = recs.index(rec)
    #                 records[category][index] = (0, rec)
    #         records_2 = season.rank_seasonal(uid, records, location)
    #         records_3 = transport.rank_transport(uid, records_2, location)
    #         to_return = format_records(records_3)
    #     else:
    #         to_return = format_records({})
    #     return to_return

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
