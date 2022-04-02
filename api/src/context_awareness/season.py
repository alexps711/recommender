from datetime import date, datetime
import os
from fetch import get_weather
from geopy.geocoders import Nominatim

WARM_MONTHS = [4, 5, 6, 7, 8, 9]
BAD_WEATHER = ['rain', 'drizzle', 'thunderstorm', 'snow']
HIGH_TEMPERATURE_CUTOFF = 20  # degrees celcius
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
geolocator = Nominatim(user_agent="coffee")


def rank_seasonal(uid, records, location):
    current_lat = location['lat']
    current_lon = location['lon']
    is_warm_month: bool = datetime.now().month in WARM_MONTHS
    location_str = str(current_lat) + ',' + str(current_lon)
    current_location = geolocator.reverse(
        location_str, language="en", addressdetails=True).raw['address']
    current_city = ""
    if "city" in current_location.keys():
        current_city = current_location["city"]
    elif "borough" in current_location.keys():
        current_city = current_location["borough"]
    elif "town" in current_location.keys():
        current_city = current_location["town"]

    current_weather, current_temperature = get_weather(
        current_lat, current_lon)

    

    for category, _ in records.items():
        for event in records[category]:
            event_date = datetime.strptime(event[1]['date'], DATE_FORMAT)
            points_for_indoors = 1 if event_date not in WARM_MONTHS else 0
            if event_date == datetime.today:
                points_for_indoors += 1 if current_temperature >= HIGH_TEMPERATURE_CUTOFF else 0
                if current_weather in BAD_WEATHER:
                    records[category][records[category].index(event)] = (event[0] + 1 + points_for_indoors if event[1]['indoors'] else 0, event[1])
                else:
                    records[category][records[category].index(event)] = (event[0] + 1 if not event[1]
                             ['indoors'] else 0, event[1])
    return records
