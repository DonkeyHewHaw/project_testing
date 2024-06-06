import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests 
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import re
from datetime import date
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
import textwrap

#Google Reviews Scraping 
def get_dynamic_user_agent():
    ua = UserAgent()
    return ua.random

def extract_place_id(url):
    # Define the regex pattern to match the place ID
    pattern = r'0x[0-9a-f]+:0x[0-9a-f]+'
    
    # Search for the pattern in the URL
    match = re.search(pattern, url)
    
    # If a match is found, return it; otherwise, return None
    if match:
        return match.group(0)
    else:
        return None

def extract_place_name(url):
    match = re.search(r'/place/([^/@]+)', url)
    if match:
        place_name = match.group(1)
        # Replace '+' with ' '
        place_name_with_spaces = place_name.replace('+', '_')
    
    return place_name_with_spaces

def save_to_txt(filename, place_name, users, line_length=80):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"LOCATION: {place_name}\n")
        f.write(f"RUN DATE: {date.today()}\n")
        f.write(f"NO. COMMENTS: {len(users)}\n\n")
        
        for user_data in users:
            f.write(f"Name: {user_data['name']}\n")
            f.write(f"Rating: {user_data['rating']}\n")
            f.write(f"Review Date: {user_data['review_date']}\n")
            review_wrapped = textwrap.fill(f"Review: {user_data['review']}", width=line_length)
            f.write(review_wrapped + '\n')
            f.write("--------------\n")


def get_reviews_data(place_review_link):
    
    headers = {
        "User-Agent": get_dynamic_user_agent()
    }

    place_name = extract_place_name(place_review_link)
    place_id = extract_place_id(place_review_link)
    
    user = []
    next_page_token = ''

    while True: 
        url = f"https://www.google.com/async/reviewDialog?hl=en_us&async=feature_id:{place_id},next_page_token:{next_page_token},sort_by:qualityScore,start_index:,associated_topic:,_fmt:pc"

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
            
        for el in soup.select('.gws-localreviews__google-review'):
            user.append({
                'name': el.select_one('.TSUbDb').text.strip(),
                'rating': el.select_one('.lTi8oc')['aria-label'],
                'review': el.select_one('.Jtu6Td').text.strip(),
                'review_date': el.select_one('.dehysf').text.strip()
            })

        next_page_token_element = soup.select_one('.gws-localreviews__general-reviews-block')
        if next_page_token_element['data-next-page-token'] != '':
            next_page_token = next_page_token_element['data-next-page-token']
        else:
            break
    
    save_to_txt(f"reviews.txt", place_name, user)

#Make Chatbot 
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text_from_txt_file(file_name):

    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, file_name)

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    cleaned_text = text.replace('\n', ' ')
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
    
    return cleaned_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question. You can give the summarization and provide recommendation according to the question if user asked.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Google Reviews Advisor")
    st.header("Get Customer Reviews' Summarization and Suggestion from Gemini")

    user_question = st.text_input("Ask a Question about Reviews")

    with st.sidebar:
        st.title("Place the URL🐆:")
        google_map_url = st.text_input("Copy and Paste Google Review URL")
        #place_name = extract_place_name(google_map_url)
    
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                get_reviews_data(google_map_url)
                raw_text = get_text_from_txt_file(f'reviews.txt')
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

            current_directory = os.getcwd()
            file_path = os.path.join(current_directory, f'reviews.txt')
            with open(file_path, 'r', encoding='utf-8') as file:
                if st.download_button(label="Download reviews as .txt file",
                                    data=file,
                                    file_name='reviews.txt',
                                    mime="text"):
                    with st.spinner("Downloading..."):
                        st.success("Download complete")
    
            
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
