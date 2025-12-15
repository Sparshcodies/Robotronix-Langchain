from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://sparshsahu567_db_user:7Sv9Gjm0V7ZNzSvO@learncluster.useiw0y.mongodb.net/?appName=learnCluster"
client = MongoClient(uri, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)