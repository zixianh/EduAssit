from pypdf import PdfReader
import os
os.environ["OPENAI_API_KEY"] = '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'
import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain import HuggingFacePipeline
from langchain_community.llms import LlamaCpp
from data_collection import create_connection, create_table, insert_data, query_data
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
import uuid
import pickle

from youtube_transcript_api import YouTubeTranscriptApi
import tensorflow_hub as hub

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from audio_recorder_custom import audio_recorder
import speech_recognition as sr


import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


st.set_page_config(page_title='TA for DSCI 552: Machine Learning', layout='wide')
st.title("EduAssist 🤖")
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)


username, authentication_status, username = authenticator.login()

if authentication_status == False:
    st. error ("Username/password is incorrect")
if authentication_status == None:
    st.warning("Please enter your username and password")
if authentication_status:

        st.markdown(" > Welcome! I am EduAssit. You could ask me anything.")

        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style.css")

        # Convert audio bytes to text
        def audio_to_text(audio_bytes, sample_width = 4):
            recognizer = sr.Recognizer()
            audio_data = sr.AudioData(audio_bytes, sample_rate=44100, sample_width=sample_width)
            try:
                text = recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return "Speech recognition could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results from Google Speech Recognition service; {e}"


        def get_button_label(chat_df):
            first_message = chat_df[(chat_df["role"] == "user")].iloc[0]["content"]
            return f"{' '.join(first_message.split()[:5])}..."


        def new_chat():
            """
            Clears session state and starts a new chat.
            """
            
            st.session_state.buffer_memory = None
            st.session_state.conversation_chain = None
            st.session_state.audio_bytes = None
            st.session_state.messages = {}


            return True
            

                    
        def get_text_chunks(text,separator,chunk_size,chunk_overlap):
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
        )

            chunks = text_splitter.split_text(text)
            return chunks

        def get_vectorstore(text_chunks):
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore


        def search_youtube_videos(query):
            API_KEY = 'AIzaSyDl7j1weHHwUzt4Grfosgr45B48T6ZErok'
            YOUTUBE_API_SERVICE_NAME = 'youtube'
            YOUTUBE_API_VERSION = 'v3'
            youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
            
            try:
                search_response = youtube.search().list(
                    q=query,
                    part='snippet',
                    maxResults=10
                ).execute()

                for search_result in search_response.get('items', []):
                    if search_result['id']['kind'] == 'youtube#video':
                        video_info = {
                            'videoId': search_result['id']['videoId'],
                            'title': search_result['snippet']['title'],
                            'description': search_result['snippet']['description'],
                            'url':'https://www.youtube.com/watch?v='+search_result['id']['videoId'],
                            'date':search_result['snippet']['publishedAt']
                        }
                        return video_info

            except HttpError as e:
                print('An HTTP error {} occurred:\n{}'.format(e.resp.status, e.content))
            return None


        def main():
            
            with st.sidebar:
                scol1, scol2 = st.columns([6,1])
                with scol1:
                    if st.button("New Chat"):
                        new_chat()
                        st.session_state.messages = {}

                with scol2:
                     st.session_state["audio_bytes"] = audio_recorder()
                    
                     if "last_bytes" in st.session_state and st.session_state.last_bytes == st.session_state.audio_bytes:
                        st.session_state.audio_bytes = None
                    

            if "model" not in st.session_state:
                st.session_state.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            if "last_bytes" not in st.session_state:
                        st.session_state["last_bytes"] = None

            if "messages" not in st.session_state:
                st.session_state.messages = {}

            for sid in st.session_state.messages.keys():
                
                if st.session_state.messages[sid] == []:
                    continue

                with st.chat_message("user"):
                    if st.session_state.messages[sid][0]["role"] == "user" and st.session_state.messages[sid][0]["type"] == "text":
                        st.markdown(st.session_state.messages[sid][0]["content"])

                with st.chat_message("EduAssist"):

                    image_found = False
                    image_if_found = []
                    curr_columns = st.columns(4)
                    #each_kind refers to one of the output by assistant: [text, images, video]
                    for media_type in range(1,len(st.session_state.messages[sid])): #One particular session
                
                            if st.session_state.messages[sid][media_type]["type"] == "text":
                                st.markdown(st.session_state.messages[sid][media_type]["content"])

                            elif st.session_state.messages[sid][media_type]["type"] == "image":
                                curr_columns[0].image(st.session_state.messages[sid][media_type]["content"][0], caption=st.session_state.messages[sid][media_type]["content"][1])

                            elif st.session_state.messages[sid][media_type]["type"] == "video":
                                curr_columns[1].video(st.session_state.messages[sid][media_type]["content"])
                            

            if ("buffer_memory" not in st.session_state) or (st.session_state.buffer_memory is None):
                st.session_state.buffer_memory = ConversationBufferMemory(
                memory_key='chat_history', return_messages=True)

    

            if "images_rows" not in st.session_state:
                # Connect to the SQLite database
                conn2 = sqlite3.connect('images.db')
                cursor2 = conn2.cursor()
                cursor2.execute("SELECT image_path, description_text FROM Image_Text")  # Update 'your_table_name' with your actual table name
                st.session_state.images_rows = cursor2.fetchall()
                st.session_state.description_embeddings = st.session_state.model.encode([row[1] for row in st.session_state.images_rows])
                
            if "vectors" not in st.session_state:
                db_file = "mydatabase.db"
                connection = create_connection(db_file)
                
                dataGet = query_data(connection)  # dataGet is a dict, key is the PDF name, value is the text of PDF file
                all_text = ''.join(dataGet.values())
                chunks = get_text_chunks(all_text,[" ", ",", "\n"], 1000, 10)
                st.session_state.vectors = get_vectorstore(chunks)

            if ("conversation_chain" not in st.session_state) or (st.session_state.conversation_chain is None):
                llm = ChatOpenAI()
                st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectors.as_retriever(
                        search_type="similarity", search_kwargs={"k": 4}),
                    memory=st.session_state.buffer_memory,
                )
        
            session_id = uuid.uuid4()
            st.session_state.messages[session_id] = []

            # React to user input
            if user_input:= st.chat_input("> Message EduAssist") or st.session_state.audio_bytes:

                    if st.session_state.audio_bytes:
                        st.session_state.last_bytes = st.session_state.audio_bytes
                        user_input = audio_to_text(st.session_state.audio_bytes)
                        st.session_state.audio_bytes = None
                    
                    st.session_state.messages[session_id].append({"role": "user", "type":"text","content": user_input})

                    # Display user input in chat message container
                    st.chat_message("user").markdown(st.session_state.messages[session_id][-1]["content"])

                    chat_history = st.session_state.conversation_chain.invoke(user_input)['chat_history'] 
                    bot_response = chat_history[-1].content if chat_history else "I'm sorry, I don't understand."
              
                    #Confirming the integrity 
                    integrity_llm = ChatOpenAI()
                    chat_history_integrity = integrity_llm.invoke(f"Analyse this response: '{bot_response} '. Is this response related to machine learning. Strictly answer in one word only: either 'yes' or 'no' only.")
         
                    with st.chat_message("EduAssist"):

                            #bot response
                            st.session_state.messages[session_id].append({"role": "assistant", "type":"text","content": bot_response})
                            st.markdown(st.session_state.messages[session_id][-1]["content"])

                            if "I'm sorry, I don't understand." not in bot_response:

                                # Compute BERT embeddings
                                question_embedding = st.session_state.model.encode([user_input])[0]
                                
                                if chat_history_integrity.content.lower() == "yes":
                                    # Compute similarity scores
                                    similarity_scores = []
                                    for row, description_embedding in zip(st.session_state.images_rows, st.session_state.description_embeddings):
                                        similarity_score = cosine_similarity(question_embedding.reshape(1, -1), description_embedding.reshape(1, -1))[0][0]
                                        similarity_scores.append((row[0], row[1], similarity_score))
                                        
                                    # show the most relevant image
                                    image_found = False
                                    max_row = max(similarity_scores, key=lambda x: x[2])

                                    columns = st.columns(4)
                                    if max_row[2] > 0.3:
                                        st.session_state.messages[session_id].append({"role": "user", "type":"image","content": (max_row[0],max_row[1])})
                                        columns[0].image(st.session_state.messages[session_id][-1]["content"][0], caption=st.session_state.messages[session_id][-1]["content"][1])
                                        image_found = True

                                    video_info = search_youtube_videos(user_input)
                                    if video_info:

                                        st.session_state.messages[session_id].append({"role": "user", "type":"video","content": video_info["url"]})
                                        if image_found == True:
                                            columns[1].video(st.session_state.messages[session_id][-1]["content"])
                                            columns[1].caption(f"Date Posted:{video_info['date']}")
                                        else:
                                            columns[0].video(st.session_state.messages[session_id][-1]["content"])
                                            columns[0].caption(f"Date Posted:{video_info['date']}")


                                    
                                        # print('Video ID:', video_info['videoId'])
                                        # print('Title:', video_info['title'])
                                        # print('Description:', video_info['description'])
                                        # print('Url:',video_info['url'])
                                        # print('Date',video_info['date'])

        if __name__ == "__main__":
            main()