import { MongoClient, ServerApiVersion } from 'mongodb';

const uri = "mongodb+srv://mongodb:12345@vector-search.tolhc.mongodb.net/?retryWrites=true&w=majority&appName=Vector-Search";

const client = new MongoClient(uri, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  }
});

export async function getIncidents() {
  try {
    await client.connect();
    const database = client.db("ChatMIM");
    const collection = database.collection("Incidents");
    
    // Fetch all documents from the collection
    const incidents = await collection.find({}).toArray();
    return incidents;
  } catch (error) {
    console.error('MongoDB Error:', error);
    throw error;
  }
}