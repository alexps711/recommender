from constants import PPRINT
import os
import requests
from helpers import convert_to_tuples
from typing import List, Dict
import json

API_URL = os.getenv('API_URL')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
TRANSPORT_API_KEY = os.getenv('TRANSPORT_API_KEY')
TRANSPORT_APP_ID = os.getenv('TRANSPORT_APP_ID')
TEMPERATURE_UNIT = 'metric'

WEATHER_CONDITION_CODES = {
    'thunderstorm': [200, 201, 202, 210, 211, 212, 221, 230, 231, 232],
    'drizzle': [300, 301, 302, 310, 311, 312, 313, 314, 321],
    'rain': [500, 501, 502, 503, 504, 511, 520, 521, 522, 531],
    'snow': [600, 601, 602, 611, 612, 613, 615, 616, 620, 621, 622],
    'clear': [800],
    'cloudy': [801, 802, 803, 804]
}  # https://openweathermap.org/weather-conditions

user = {}


def fetch_categories():
    """ Return the full list of categories
    """
    r = requests.get(API_URL + '/coffee/categories')
    categories = r.json()['params']
    return categories


def fetch_user_categories(uid):
    """ Return user's selected categories.

        :param uid: the user id.
    """
    payload = {'uid': uid}
    r = requests.post(API_URL + '/coffee/categories', data=payload)
    categories = r.json()['params']
    return categories


def fetch_user_optional_data(uid):
    """ Return user's preferences input during onboarding.

        :param uid: the user id.
    """
    try:
        payload = {'uid': uid}
        r = requests.post(API_URL + '/coffee/optional', data=payload)
        optional_data = r.json()['params']
        if optional_data:
            return optional_data
        else:
            raise Exception
    except:
        return None  # no data found.


def fetch_records_by_categories(matches, location):
    """ Return records based on given, matched categories.

        :param matches: a list of categories.
        : param location: the user's location.
    """
    payload = {'matches': matches, 'location': json.dumps(location)}
    r = requests.post(API_URL + '/coffee/records', data=payload)
    records = r.json()['params']
    return records


def fetch_prev_purchases(uid) -> Dict[str, list]:
    payload = {'uid': uid}
    r = requests.post(API_URL + '/coffee/history', data=payload)
    purchases = r.json()['params']
    return purchases


def get_user_nearby_friends_events(uid, location) -> list:
    payload = {'uid': uid, 'location': json.dumps(location)}
    response = requests.post(API_URL + '/coffee/friends', data=payload)
    nearby_friend_events = response.json()['params']
    nearby_friend_events = convert_to_tuples(nearby_friend_events)
    return nearby_friend_events


def get_friends_prev_events(uid, location):
    payload = {'uid': uid, 'location': json.dumps(location)}
    r = requests.post(API_URL + '/coffee/friends?past=true', data=payload)
    friends_prev_events = r.json()['params']
    friends_prev_events = convert_to_tuples(friends_prev_events)
    return friends_prev_events
    # return [({'id': i, 'category': 'restaurants', 'cuisine': 'italian', 'date': '2018-12-25 00:00:00', 'indoors': True, 'lat': 51.513884, 'lon': -0.153889}, i) for i in range(100)]


def get_nearby_users(uid, location, radius):
    """ Fetch users nearby (1000 limit).

        :param uid the user id.
        :param location the user's location.
        :param radius the area to search users in (in miles).
    """
    payload = {'uid': uid, 'location': json.dumps(location), 'radius': radius}
    r = requests.post(API_URL + '/coffee/nearby', data=payload)
    nearby_users: List[str] = r.json()['params']
    return nearby_users
    # return [i for i in range(50)]


def get_users_future_events(uids: list):
    payload = {'uids': json.dumps(uids)}
    r = requests.post(API_URL + '/coffee/events', data=payload)
    events_response = r.json()['params']
    events_final = {category: convert_to_tuples(
        events) for category, events in events_response.items()}
    return events_final


def get_weather(lat, lon, id=0):
    r = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units={TEMPERATURE_UNIT}")
    weather = r.json()
    weather_code = weather['weather'][0]['id']
    temperature = weather['main']['temp']
    weather_condition = None
    for key, value in WEATHER_CONDITION_CODES.items():
        if weather_code in value:
            weather_condition = key
    return weather_condition, temperature


def get_nearby_tube_stations(bbox: dict):
    min_lat = bbox['min_lat']
    min_lon = bbox['min_lon']
    max_lat = bbox['max_lat']
    max_lon = bbox['max_lon']
    r = requests.get(
        f"http://transportapi.com/v3/uk/tube/stations/bbox.json?minlon={min_lon}&minlat={min_lat}&maxlon={max_lon}&maxlat={max_lat}&page=1&rpp=10&app_id={TRANSPORT_APP_ID}&app_key={TRANSPORT_API_KEY}").json()
    return r['stations']


def search_by_tokens(tokens: list, location: dict, filters: dict):
    payload = {'tokens': json.dumps(tokens, default=str), 'location': json.dumps(location), 'filters': json.dumps(filters)}
    r = requests.post(API_URL + '/coffee/search', data=payload)
    result = r.json()['params']
    return result
