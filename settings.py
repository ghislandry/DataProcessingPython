import os
# __file__ refers to the file settings.py 
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')
ML_DATA_FILE = "Indonesia 6.csv" # The file should reside in the 'static' directory
