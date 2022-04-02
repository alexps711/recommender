from typing import Dict
from fetch import get_user_nearby_friends_events, get_friends_prev_events, get_nearby_users, get_users_future_events, fetch_user_categories

# percentage of total users going to an event to be considered trendy
TRENDING_EVENT_METRIC = 0.15
# diameter in miles to define the search area for users
RADIUS = 10


def get_nonfriends_nearby_events(uid, location):
    nearby_users = get_nearby_users(uid, location, radius=RADIUS)
    raw_events = get_users_future_events(nearby_users)
    for category, events in raw_events.items():
        raw_events[category] = list(filter(lambda kv: kv[1] > 1, events))
    return len(nearby_users), raw_events


def rank_collective(uid, records: Dict[str, list], location):
    nearby_events = get_user_nearby_friends_events(uid, location)
    friends_prev_events = get_friends_prev_events(uid, location)
    friends_prev_events_list = [event for event,
                                friends_going in friends_prev_events]
    # join to be able to only do one iteration
    joint_events = nearby_events + friends_prev_events
    for event, friends_going in joint_events:
        if any(record_event == event for points, record_event in records[event['category']]):
            if friends_going > 1:
                map(lambda points, record_event: (points + 15 if event in friends_prev_events_list else 10,
                                                  event) if record_event == event else (points, event), records[event['category']])
            else:
                map(lambda points, record_event: (points + 5, event) if record_event ==
                    event else (points, event), records[event['category']])
        else:
            records[event['category']].append(
                (8 if event in friends_prev_events and friends_going > 1 else 5, event))  # add event to dataset
    number_of_users, nearby_events = get_nonfriends_nearby_events(
        uid, location)  # look at nearby users future events now
    for category, events in nearby_events.items():
        for event, users_going in events:
            if any(record_event == event for points, record_event in records[category]):
                # !!! get user categories from some other place to avoid calling api twice.
                map(lambda points, record_event: (points + 5 if event[category] in fetch_user_categories(uid) else 3, record_event)
                    if record_event == event else (points, record_event), records[category])
            else:
                if users_going > number_of_users*TRENDING_EVENT_METRIC:
                    map(lambda points, record_event: (points + 20, record_event)
                        if record_event == event else (points, record_event), records[category])
                else:
                    records[category].append((20, event))
    return records