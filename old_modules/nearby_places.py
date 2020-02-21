from googleplaces import GooglePlaces, types, lang 
import requests 
import json
from geopy.distance import great_circle
from gtts import gTTS
import os
import geocoder
def get_Nearest_Petrol_Station(current_location):
  #API_KEY = 'AIzaSyBwd1ekYqgRLNSOvs9TOr5E1aEcdEoyiIs'
  API_KEY = 'AIzaSyCNya2vpVocmsGbhG5BIA2KtyzdvNH0qmQ'
  google_places = GooglePlaces(API_KEY)


  query_result = google_places.nearby_search(lat_lng ={'lat':current_location[0], 'lng':current_location[1]},radius = 5000,types =[types.TYPE_GAS_STATION]) 

  if query_result.has_attributions: 
      print (query_result.html_attributions) 
  dictionary = {}

  for i,place in enumerate(query_result.places):
    dictionary[place.name] = (float(place.geo_location['lat']),float(place.geo_location['lng']))   
  smallest_distance = 1000
  nearest = ()
  for i in dictionary.values():
    temp = great_circle(current_location,i).km
    if temp<smallest_distance:
      smallest_distance = temp
      nearest = i
  print(nearest)
  x = [place for place,cordinates in dictionary.items() if cordinates==nearest]
  print('Nearest Petrol pump is',"".join(x),'at', smallest_distance,'Km')
  tts = gTTS(text='Nearest Petrol pump is'+"".join(x)+'at'+str(smallest_distance)[0:5]+'Km', lang='en')
  tts.save("output.mp3")
  os.system("mpg321 output.mp3")#Text to speach
#current = (22.7177,75.8586)#current longitude and latitude data to be collected from gps device.
# t = (g.latlng[0],g.latlng[1])
# get_Nearest_Petrol_Station(t)

