import data_parser as parser
import pandas as pd
import numpy as np
import helpers as helper

def get_user_events(user_id):
    """Get the events that a user has attended.

    Args:
        user_id (int): the id of the user.

    Returns:
        list: A list of the event ids the user has attended.
    """
    events = []
    attendees = parser.get_event_attendees() 
    # take only users that have actually attended an event
    yes_attendees = attendees.yes.dropna()
    np_attendees = yes_attendees.apply(lambda x: x.split(' '))
    # look for the user in the list of attendees of every event
    arr = np.array(np_attendees)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == user_id:
                events.append(attendees.iloc[i].name)
    return events

def get_location_similarity(user_id, event_id):
    user = helper.get_user(user_id)
    event = helper.get_event(event_id)
    # TODO: implement
                

if __name__ == "__main__":
    get_user_events(1975964455)    
