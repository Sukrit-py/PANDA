import os
import google.generativeai as genai
import pandas as pd
import clickhouse_connect
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Google Gemini API configuration
genai.configure(api_key="------------")

# Load the PDF and split it into pages
loader = PyPDFLoader("")
pages = loader.load_and_split()
pages = pages[1:]  # Skip unnecessary pages
text = "\n".join([doc.page_content for doc in pages])

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.create_documents([text])
for i, d in enumerate(docs):
    d.metadata = {"doc_id": i}

# Function to generate embeddings for a text
def get_embeddings(text):
    model = 'models/embedding-001'
    embedding = genai.embed_content(model=model, content=text, task_type="retrieval_document")
    return embedding['embedding']

# Generate embeddings for the document content
content_list = [doc.page_content for doc in docs]
embeddings = [get_embeddings(content) for content in content_list]

# Create a dataframe with page content and embeddings
dataframe = pd.DataFrame({
    'page_content': content_list,
    'embeddings': embeddings
})

# Clickhouse/MyScaleDB connection
client = clickhouse_connect.get_client(
    host='',
    port= ,
    username='',
    password=''
)

# Check if the table exists, create if not
table_exists = client.command("""
    SELECT count(*) 
    FROM system.tables 
    WHERE database = 'default' AND name = 'handbook'
""")

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

# Insert data into ClickHouse in batches
batch_size = 10
num_batches = len(dataframe) // batch_size
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_data = dataframe[start_idx:end_idx]
    client.insert("default.handbook", batch_data.to_records(index=False).tolist(), column_names=batch_data.columns.tolist())
    print(f"Batch {i+1}/{num_batches} inserted.")

# Check if the vector index exists, create if not
index_query_result = client.query("SELECT name FROM system.vector_indices WHERE table='handbook' AND name='vector_index'")

if not index_query_result.result_set:
    client.command("""
    ALTER TABLE default.handbook
        ADD VECTOR INDEX vector_index embeddings
        TYPE MSTG
    """)
    print("Vector index created.")
else:
    print("Vector index already exists.")
