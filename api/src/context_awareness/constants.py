import pathlib
import pprint
import os

PATH = pathlib.Path().absolute()
PPRINT = pprint.PrettyPrinter(indent=4)
MODEL_PATH = str(PATH) + "/nlp/model"
DATASETS = ['london_boroughs', 'dates', 'events']
class ENTITIES:
    EVENT = 'EVENT'
    LOCATION = 'LOC'
    DATE = 'DATE'
class USER:
    ID = None
    CATEGORIES = None
