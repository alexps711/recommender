import collections
import functools
import data_parser as parser


class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)
    
def get_user(user_id):
    """Get a user from the data.

    Returns:
        pandas.DataFrame: The user object
    """
    users = parser.get_users()
    return users[users.user_id == user_id]

def get_event(event_id):
    """Get an event from the data.

    Returns:
        pandas.DataFrame: The event object
    """
    events = parser.get_events()
    return events[events.event_id == event_id]
