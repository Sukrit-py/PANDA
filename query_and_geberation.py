import google.generativeai as genai
import clickhouse_connect

# Google Gemini API configuration
genai.configure(api_key="")

# Clickhouse/MyScaleDB connection
client = clickhouse_connect.get_client(
    host='',
    port= ,
    username='',
    password=''
)

# Function to generate embeddings for a text query
def get_embeddings(text):
    model = 'models/embedding-001'
    embedding = genai.embed_content(model=model, content=text, task_type="retrieval_document")
    return embedding['embedding']

# Function to retrieve relevant documents based on user query
def get_relevant_docs(user_query):
    query_embeddings = get_embeddings(user_query)
    results = client.query(f"""
        SELECT page_content,
        distance(embeddings, {query_embeddings}) as dist FROM default.handbook ORDER BY dist LIMIT 3
    """)
    relevant_docs = [row['page_content'] for row in results.named_results()]
    return relevant_docs

# Function to create the prompt for RAG
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

# Function to generate a response using the RAG setup
def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text

# Main function to handle query and response
def generate_answer(query):
    relevant_text = get_relevant_docs(query)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    answer = generate_response(prompt)
    return answer

# Example usage
if __name__ == "__main__":
    answer = generate_answer(query="I want to know about Schizophrenia")
    print(answer)
