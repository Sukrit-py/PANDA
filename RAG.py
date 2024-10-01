import os
import google.generativeai as genai
import pandas as pd
import clickhouse_connect
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")



loader = PyPDFLoader("PANDA-main/Docx/psychological_symptoms.pdf")
pages = loader.load_and_split()
pages = pages[1:]  # Skip the first few pages as they are not required
text = "\n".join([doc.page_content for doc in pages])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.create_documents([text])
for i, d in enumerate(docs):
    d.metadata = {"doc_id": i}



# This function takes a a sentence as an arugument and return it's embeddings
def get_embeddings(text):
    # Define the embedding model
    model = 'models/embedding-001'
    # Get the embeddings
    embedding = genai.embed_content(model=model,
                                    content=text,
                                    task_type="retrieval_document")
    return embedding['embedding']
# Get the page_content from the documents and create a new list
content_list = [doc.page_content for doc in docs]
# Send one page_content at a time
embeddings = [get_embeddings(content) for content in content_list]

# Create a dataframe to ingest it to the database
dataframe = pd.DataFrame({
    'page_content': content_list,
    'embeddings': embeddings
})



client = clickhouse_connect.get_client(
    host='-------',
    port=---,
    username='-------------',
    password='--------------'
)


# Check if the table exists in ClickHouse
table_exists = client.command("""
    SELECT count(*) 
    FROM system.tables 
    WHERE database = 'default' AND name = 'handbook'
""")

# Create the table only if it doesn't exist
if table_exists == 0:
    client.command("""
        CREATE TABLE default.handbook (
            id Int64,
            page_content String,
            embeddings Array(Float32),
            CONSTRAINT check_data_length CHECK length(embeddings) = 768
        ) ENGINE = MergeTree()
        ORDER BY id
    """)
    print("Table 'handbook' created.")
else:
    print("Table 'handbook' already exists.")

# Insert the data in batches
batch_size = 10
num_batches = len(dataframe) // batch_size
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_data = dataframe[start_idx:end_idx]
    # Insert the data
    client.insert("default.handbook", batch_data.to_records(index=False).tolist(), column_names=batch_data.columns.tolist())
    print(f"Batch {i+1}/{num_batches} inserted.")


# Check if the vector index exists
index_query_result = client.query("SELECT name FROM system.vector_indices WHERE table='handbook' AND name='vector_index'")

if index_query_result.result_set:  # Check if there are results
    index_exists = True
else:
    index_exists = False

if not index_exists:
    client.command("""
    ALTER TABLE default.handbook
        ADD VECTOR INDEX vector_index embeddings
        TYPE MSTG
    """)
    print("Vector index created.")
else:
    print("Vector index already exists.")



def get_relevant_docs(user_query):
    # Call the get_embeddings function again to convert user query into vector embeddings
    query_embeddings = get_embeddings(user_query)
    # Make the query to fetch relevant documents
    results = client.query(f"""
        SELECT page_content,
        distance(embeddings, {query_embeddings}) as dist FROM default.handbook ORDER BY dist LIMIT 3
    """)
    relevant_docs = []
    for row in results.named_results():
        relevant_docs.append(row['page_content'])
    
    # Drop the vector index after retrieving the relevant docs
    client.command("""
    ALTER TABLE default.handbook
        DROP VECTOR INDEX vector_index
    """)
    print("Vector index dropped.")
    
    return relevant_docs


def make_rag_prompt(query, relevant_passage):
    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"You are a psychologist named Dr.PANDA, please provide insights or advice. "
        f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
        f"Maintain a polite and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt


def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text


def generate_answer(query):
    relevant_text = get_relevant_docs(query)
    text = " ".join(relevant_text)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    answer = generate_response(prompt)
    return answer

answer = generate_answer(query="is social anxiety a anxiety disorder?")
print(answer)