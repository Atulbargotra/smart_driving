import requests
import json
from text_to_speech import speak
from location import get_location
import os
from dotenv import load_dotenv
load_dotenv()
def get_Nearest_Petrol_Station(longitude,latitude,category,radius=2000):
    API_KEY = os.getenv('HERE_MAPS_API_KEY')
    url = f"https://places.sit.ls.hereapi.com/places/v1/browse?apiKey={API_KEY}&in={latitude},{longitude};r={radius}&cat={category}&pretty"
    x = requests.get(url)
    #petrol-station
    data = json.loads(x.text)
    nearest_place = data['results']['items'][0]['title']
    distance = data['results']['items'][0]['distance']
    text_input = f"Nearest Petrol pump is {nearest_place} at {distance} meter"
    speak(text_input)