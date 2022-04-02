from pprint import pprint
from fetch import fetch_prev_purchases
from helpers import is_similar_category


def rank_history(uid, records):
    prev_purchases = fetch_prev_purchases(uid)
    sorted(prev_purchases.items(), key=len)
    purchases_categories = prev_purchases.keys()
    top_categories = list(prev_purchases.keys())[
        :2]  # 3 most purchased categories
    for category in purchases_categories:
        prev_parent_ids = [item['parent_id']
                           for item in prev_purchases[category]]
        for points, item in records[category]:
            index = records[category].index((points, item))
            if item['parent_id'] in prev_parent_ids:
                # Record the user has purchased tickets for this event.
                records[category][index] = (
                    points + 3, {**item, 'purchased': True})
            if category in top_categories:
                records[category][index] = (
                    points + 2 if is_similar_category(category, top_categories) else 1, records[category][index][1])

    return records
