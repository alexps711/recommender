import os
from fetch import get_nearby_tube_stations
from math import radians, cos, sin, asin, sqrt
from geopy import distance
from typing import List
from helpers import get_bounding_box
from constants import PPRINT

MILES_RADIUS = 5


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_great_circle_distance(point1: tuple, point2: tuple) -> float:
    return distance.great_circle(point1, point2).miles


def get_points_distance(distance: float):
    if distance < 5:
        if distance < 2:
            return 2
        else:
            return 1
    else:
        return 0


def proximity_sorting(current_location, record):
    event_location = (record[1]['lat'], record[1]['lon'])
    return get_great_circle_distance(current_location, event_location)


def get_nearby_stations(event_location: tuple, all_tube_stations: list):
    """ Return the 3 closest tube stations to a given location,
        :param event_location: the lat and lon of the event.
        :param all_tube_stations: all tube station in the bounding box that
            includes the event location.
    """
    station_distance_list = []
    for station in all_tube_stations:
        station_location = (station['latitude'], station['longitude'])
        distance_to_event = get_great_circle_distance(
            event_location, station_location)
        if distance_to_event < 500:
            station_distance_list.append(
                {'station': station, 'distance': distance_to_event})
    closest_stations = sorted(station_distance_list,
                              key=lambda station: station['distance'])[:3]
    return closest_stations


def rank_transport(uid, records, location):
    current_location = (location['lat'], location['lon'])
    # get the area that includes all events
    bouding_box = get_bounding_box(records)
    all_tube_status = get_nearby_tube_stations(bouding_box)
    for category, recs in records.items():
        recs.sort(key=lambda record: proximity_sorting(
            current_location, record))  # sort by closest distance
        # keep track of the last tube stations to stamp in events in the same cluster
        for record in recs:
            event_location = (record[1]['lat'], record[1]['lon'])
            distance_from_current_location = get_great_circle_distance(
                current_location, event_location)
            points_for_distance = get_points_distance(
                distance_from_current_location)
            tube_stations = get_nearby_stations(
                event_location, all_tube_status)
            points_to_add = points_for_distance + \
                len(tube_stations)  # revise this
            # store tube station info
            record[1]['transport'] = {
                'tube_stations': [{'name': station['station']['name'], 'distance': station['distance'], 'lines': station['station']['lines']} for station in tube_stations]
            }
            recs[recs.index(record)] = (record[0] + points_to_add, record[1])

    return records
