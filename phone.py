from twilio.rest import Client #MAIN
import os
from dotenv import load_dotenv # for storing API KEYS
load_dotenv()
def call(user_phone_no,print_sid=True):
    ACCOUNT_SID = os.getenv('ACCOUNT_SID')
    AUTH_TOKEN = os.getenv('AUTH_TOKEN')
    client = Client(ACCOUNT_SID,AUTH_TOKEN)
    call = client.calls.create(url='https://phone-72ed9.firebaseapp.com/index.xml',to=user_phone_no,from_='+16029753389')
    if print_sid:
        print(call.sid)
