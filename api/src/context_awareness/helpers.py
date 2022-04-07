from constants import PPRINT, USER
import os
import requests
from typing import List
import geocoder

def get_current_location() -> dict:
    location = geocoder.ip('me').latlng
    return {'lat': location[0], 'lon': location[1]}


def convert_to_tuples(events: list) -> list:
    """ Take a list of events and extract out the number of friends going
        This number will be the second argument of the tuple, the event being the first.

        :param events: the list of event objects.
    """
    def convert_event(event):
        number_of_friends = int(event['friends_purchases'])
        return (event, number_of_friends)
    formatted_events = map(convert_event, events)
    return list(formatted_events)


def get_bounding_box(records: dict):
    """ Return the area defined by min and max latitude and longitude
        that contains all the events in the records.

        :param records: the categorized events.
    """
    all_lats = []
    all_lons = []
    for _, recs in records.items():
        for rec in recs:
            all_lats.append(rec[1]['lat'])
            all_lons.append(rec[1]['lon'])

    return {
        'min_lat': min(all_lats),
        'max_lat': max(all_lats),
        'min_lon': min(all_lons),
        'max_lon': max(all_lons)
    }


def format_records(records: dict):
    """ Turn the records into a list, sort them by points and format data.
        :param records: the dict to format.
    """
    response = []
    for category, events in records.items():

        for points, event in events:
            try:
                response.append(
                    {
                        'rank': points,
                        'id': event['id'],
                        'parent_id': event['parent_id'],
                        'event': {
                            'category': event['category'],
                            'subcategory': event['subcategory'],
                            'date': event['date'],  # YYY-MM-DD HH:MM:SS
                            'short_description': event['short_description'],
                            'description': event['description'],
                            'title': event['title'],
                            'purchased': 'purchased' in event.keys(),
                            'friends_purchases': event['friend_purchases'] if 'friends_purchases' in event.keys() else 0,
                            'nearby_purchases': event['nearby_purchases'] if 'nearby_purchases' in event.keys() else 0
                        },
                        'venue': {
                            'id': event['venue_id'],
                            'location': {
                                'lat': event['lat'],
                                'lon': event['lon']
                            },
                            'indoors': event['indoors'],
                        },
                        'transport': event['transport']
                    }
                )
            except:
                break
    if(len(response) > 0):
        response = {
            'status': 200,
            'message': "Successfully retrieved records",
            'params': {
                'uid': USER.ID,
                'categories': USER.CATEGORIES,
                'events': response
            }
        }
    else:
        response = {
            'status': 404,
            'message': "No events were found",
            'params': {
                'uid': USER.ID,
                'categories': USER.CATEGORIES,
                'events': []
            }
        }
    return response
