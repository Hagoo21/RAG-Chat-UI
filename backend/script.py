from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "ChatMIM"  # Replace with your database name
COLLECTION_NAME = "Incidents"  # Replace with your collection name

def clear_collection():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Confirm the connection
        client.admin.command('ping')
        print(f"Connected to MongoDB database: {DB_NAME}, collection: {COLLECTION_NAME}")

        # Delete all documents from the collection
        result = collection.delete_many({})
        print(f"Deleted {result.deleted_count} documents from the collection.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the MongoDB connection
        client.close()
        print("MongoDB connection closed.")

if __name__ == "__main__":
    clear_collection()
