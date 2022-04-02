import pandas as pd
import numpy as np
from helpers import memoized
from data_parser import users, events, friends, test, attendees, train

def get_user_events(user_id):
    """Get the events that a user has attended.

    Args:
        user_id: the id of the user.
        attendees: the dataframe of attendees.
        
    Returns:
        list: A list of the event ids the user has attended.
    """
    evs = []
    # take only users that have actually attended an event
    yes_attendees = attendees.yes
    # look for the user in the list of attendees of every event
    for index, attendees_list in yes_attendees.items():
        try:
            np_attendees = np.array(attendees_list.split(' '))
        except:
            # attendees list == 'nan'
            np_attendees = np.array([])
        if str(user_id) in np_attendees:
            event = attendees.iloc[index].event_id
            evs.append(event)
    return evs

def get_friends_events(user_id):
    """Get the events that a user's friends have attended and the count.

    Args:
        user_id (int): the id of the user

    Returns:
        dict: A dictionary of the events that the user's friends have attended and the count.
    """
    events_dict = {}
    friends_raw = friends[friends.user_id == user_id].friends.dropna()
    friends_final = friends_raw.apply(lambda x: x.split(' '))
    friends_final = np.array(friends)[0]
    for friend in friends_final:
        events = get_user_events(friend)
        for event in events:
            events_dict[event] = events_dict.get(event, 0) + 1
    return events_dict

@memoized
def get_event_attendees(event_id):
    """Get the attendees of an event.

    Args:
        event_id (int): The id of the event.

    Returns:
        list: The list of attendees (user_id) of the event.
    """
    attendees_raw = attendees[attendees.event_id == event_id].yes.dropna()
    if not attendees_raw.empty:
        attendees_final = attendees_raw.apply(lambda x: x.split(' '))
        return np.array(attendees_final)[0]
    else:
        return []

def get_user_friends(user_id):
    """Get the friends of a user.

    Args:
        user_id (int): The id of the user.

    Returns:
        list: The list of friends (user_id) of the user.
    """
    user_friends = friends[friends.user_id == user_id].friends.dropna()
    if user_friends.empty:
        return []
    friends_final = user_friends.apply(lambda x: x.split(' '))
    return np.array(friends_final)[0]
    
def get_location_similarity(user_id, event_id):
    """Get the location similarity between a user and an event.
    The location similarity is computed by taking into account 
    city, state and country mentioned in the user's location.

    Args:
        user_id (int): The id of the user.
        event (DataFrame): The event

    Returns:
        int: the computed similarity between the user's location
        and the event.
    """
    user = get_user(user_id)
    event = get_event(event_id)
    if event.city.to_string() in user.location.to_string():
        return 1
    elif event.state.to_string() in user.location.to_string():
        return 0.5
    elif event.country.to_string() in user.location.to_string():
        return 0.25
    return 0

def get_age_similarity(user_id, event_attendees):
    """Get the age similarity between a user and the attendees of an event.
    Max similarity is 1, min similarity is 0.

    Args:
        user_id (int): The id of the user.
        event_id (id): The id of the event.

    Returns:
        int: the computed similarity between the user's age and the event's attendees' age.
    """
    # compute average age of event attendees
    birthyears = []
    for uid in event_attendees:
        attendee = get_user(uid)
        if not attendee.empty:
            attendee = attendee.iloc[0] # TODO: NOT HITTING
            if attendee.birthyear is not None:
                birthyears.append(int(attendee.birthyear))
    avg_birthyear = np.mean(birthyears)
    # compute age similarity
    # TODO: remove try-except after testing
    try:
        user = get_user(user_id).iloc[0]
    
        user_birthyear = int(user.birthyear)
        if avg_birthyear - 2 <= user_birthyear < avg_birthyear + 2:
            return 1
        elif avg_birthyear - 5 <= user_birthyear < avg_birthyear + 5:
            return 0.5
        return 0
    except: 
        return 0    

@memoized
def get_user(user_id):
    """Get a user from the data.

    Returns:
        pandas.DataFrame: The user object
    """
    return users[users.user_id == user_id]

@memoized
def get_event(event_id):
    """Get an event from the data.

    Returns:
        pandas.DataFrame: The event object
    """
    return events[events.event_id == event_id]

if __name__ == '__main__':
    print("This is the API module")

