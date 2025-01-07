import dspy
import os
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv
from langchain_pinecone import PineconeVectorStore
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
import google.generativeai as genai

Gemini_API_Key = os.getenv("Evoq_Gemini")
Pinecone_api = os.environ.get("PINECONE_API_KEY")
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=Gemini_API_Key)
pc = Pinecone(api_key=Pinecone_api)

index_name = "haylese"
namespace = "haylese_json"

index = pc.Index(index_name)
vector_store = PineconeVectorStore(embedding=gemini_embeddings,index=index,namespace=namespace)

with open('HaylyseSample_Json_Final.json', 'r') as f:
    input_val = json.load(f)

prompt_template = PromptTemplate(
    template= 
    "   ** You are a Travel Itinerary Generator. Your task is to generate a travel itinerary based on the 'User's requirements' and a database that includes details about traveling cities, sightseeing activities, hotels, resturents.\
        User's requirement:{query1}\
        Itinerary_output_format: {input_val}\
        ** In some 'User's requirements', there are description about number of traveling days, traveling cities and sightseeing activities, hotels, resturents.\
        ** If there is no any details about traveling cities and sightseeing activities, hotels, resturents in 'User's requirements', use the following guidelines: \
                1. Daily Structure: \
                    * Each day in the itinerary must include the 'city' and 'activity' keys. \
                    * 'activity' key must include sightseeing activities, hotels, resturents.\
                    * Add more than one sightseeing activity for a day, if the value of 'type_of_day' key does not equal to 'Rest Day'\
                2. Sightseeing Activities:\
                    * Sightseeing activities must select from the 'Locations' key in database. \
                    * Do not derive sightseeing activities that is 'location_name' key in the 'Itinerary_output_format' directly from the 'city_description' or 'Location_description' key in the database.\
                    * If a requested sightseeing activity(location) is unavailable in the database, select the next best alternative from the same city.\
                    * Ensure every activity in the itinerary has a valid 'location_id' in the 'Itinerary_output_format'. The 'location_id' must not be null.\
                    * If you select any sightseeing activity that 'location_id' equlas to NULL, remove that sightseeing activity from the travel itinerary.\
                    * Do not derive sightseeing activities directly from the 'city_description' or 'Location_description' key in the database.\
                4. Feasibility: \
                    * Analyze the feasibility of sightseeing activities within a single day. Ensure the total average visiting time for the selected sightseeings does not exceed the total possible visiting hours, which is a minimum of 10 hours per day\
                5. Maximization: \
                    * Select maximum number of sightseeing activities from the provided database per day whenever practical.\
                6. Optimization: \
                    * Choose the optimal set of sightseeing activities from the database that maximizes the experience while adhering to the practical time constraints.\
        Output Requirements:\
            ** Format: The output must strictly follow the structure and key names of the provided 'Itinerary_output_format'.\
            ** Key Names: Do not alter any key names in the 'Itinerary_output_format'.\
    Final Output: Generate a travel itinerary in JSON format strictly adhering to the above instructions and constraints. \
    "
    )
query1 = "Can you suggest a 4-day plan for exploring caves, forests, and geological wonders"


Prompt = prompt_template.format(input_val=f"{input_val}",query1=f"{query1}")
#print(Prompt)

with open("Haylese_text_Final_input_11.txt", "w") as file:
    file.write(Prompt)

output_path = r"C:\Evoq\Text_to_Json_Haylise\Text_itinerary_Final\Haylese_text_Final_Output_11.json"


class GeminiLM(dspy.LM):
    def __init__(self, model, apikey=None, endpoint=None):
        genai.configure(api_key= Gemini_API_Key or apikey)
        self.api_key = Gemini_API_Key if apikey is None else apikey
        self.endpoint = endpoint
        self.history = []

        super().__init__(model)
        self.model = genai.GenerativeModel(model)

    def __call__(self, prompt=None, messages=None):
        # Custom chat model working for text completion model
        prompt = '\n\n'.join([x['content'] for x in messages] + ['BEGIN RESPONSE:'])

        completions = self.model.generate_content(prompt)
        self.history.append({"prompt": prompt, "completions": completions})

        # Must return a list of strings
        return [completions.candidates[0].content.parts[0].text]

    def inspect_history(self):
        for interaction in self.history:
            print(f"Prompt: {interaction['prompt']} -> Completions: {interaction['completions']}")

    def __reduce__(self):
        return (GeminiLM, (self.model, self.api_key, self.endpoint), self.__dict__)
    
gemini = GeminiLM("gemini-1.5-flash")

class Passage:
    def __init__(self, content):
        self.long_text = content  # The expected attribute by DSPy

def retrieval_function(query, k=10, namespace=namespace):
    # Perform similarity search in the vector store
    retrieved_docs = vector_store.similarity_search(query=query, k=k, namespace=namespace)
    # Return the passages with the required structure
    return [Passage(doc.page_content) for doc in retrieved_docs]


dspy.settings.configure(lm=gemini,rm= retrieval_function)

class RAGSignature(dspy.Signature):
    context = dspy.InputField(desc="Relevant context retrieved from given databse for the User's requirement")
    question = dspy.InputField(desc="The User's requirement")
    response = dspy.OutputField(desc="The generated response based on the User's requirement and context as key value pairs")


class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(RAGSignature)

    def forward(self, question):
        retriever = dspy.Retrieve(k=10)
        context = retriever(query1).passages
        
        return self.respond(context=context, question=question)

SummarizeModule_func = RAG()


response = SummarizeModule_func(question = Prompt)
print(response)
response = response["response"]
response = response.strip('```json\n').strip('```')
print(response)
print(type(response))
response_json = json.loads(response)
print(response_json)
print(type(response_json))

with open(output_path, "w") as file:
    json.dump(response_json,file, indent=4)
