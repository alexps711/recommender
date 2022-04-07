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
