from constants import USER
from helpers import is_similar_category
from fetch import *
import sys
sys.path.append("../")


def get_category_matching(category, user_categories):
    """ Assign points to the category based on it totally or partially
        matching user preferences.
    """
    if category not in user_categories:
        return {category: 2} if is_similar_category(category, user_categories) else {category: 1}
    else:
        return {category: 3}


def get_category_points(category, user_categories):
    """ Return the points each category deserves.

        :param category: the category to give points to.
        :param user_categories: the user's favourite categories.
    """
    partial_matches = []  # to be updated
    if category in user_categories:
        return 3
    elif category in partial_matches:
        return 2
    else:
        return 1


def get_categories_with_points(category, user_categories, unordered_records):
    """ Turn records into tuples with their appropiate points

        :param category: the category to filter the records by.
        :param user_categories: the categories the user selected during onboarding.
        :param unordered_records: the list of fetched records without specific order.
    """
    records = list(filter(lambda record: record['category'] ==
                          category, unordered_records))
    return list(map(lambda record: (get_category_points(category, user_categories), record), records))


def assign_points_optional_data(optional_data, ordered_records):
    """ Give points to record's that match criteria from optional data inputs.

        :param optional_data: the optional data input from the user during onboarding.
        :param ordered_records: fetched records ordered by category.
    """
    for option in optional_data:
        category = option['category']
        subcategory = option['subcategory']
        preference = option['preference']
        for points, event in ordered_records[category]:
            try:
                if event[subcategory] == preference:
                    index = ordered_records[category].index((points, event))
                    points += 1
                    # replace tuple with updated points as they cannot be altered
                    ordered_records[category][index] = (points, event)
            except: 
                print("error")


def rank_explicit(uid, location):
    """ Rank records using user's directly input data.

        :param uid: the user's id.
        :param location: the user's current location coordinates.
    """
    all_categories = fetch_categories()
    user_categories = fetch_user_categories(uid)
    USER.CATEGORIES = user_categories # save user categories
    matches = [get_category_matching(category, user_categories)
               for category in all_categories]
    unordered_records = fetch_records_by_categories(matches, location)
    ordered_records = {category: get_categories_with_points(category, user_categories, unordered_records)
                       for category in all_categories
                       }
    optional_data = fetch_user_optional_data(uid)
    assign_points_optional_data(optional_data, ordered_records)
    return ordered_records
