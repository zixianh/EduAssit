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

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_text_chunks(text,separator,chunk_size,chunk_overlap):
    # text_splitter = CharacterTextSplitter(
    #     separator=separator,
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    
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

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain


def main():

    
    db_file = "mydatabase.db"
    connection = create_connection(db_file)
    
    dataGet = query_data(connection)  # dataGet is a dict, key is the PDF name, value is the text of PDF file
    all_text = ''.join(dataGet.values())
    chunks = get_text_chunks(all_text,[" ", ",", "\n"], 1000, 10)
    vectors = get_vectorstore(chunks)
    conversation_chain = get_conversation_chain(vectors)
    
    # Connect to the SQLite database
    conn2 = sqlite3.connect('images.db')
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT image_path, description_text FROM Image_Text")  # Update 'your_table_name' with your actual table name
    rows = cursor2.fetchall()
    
    print("Welcome! I am EduAssit. You could ask me anything. Type 'exit' to quit.")

    question_count = 0
    while True:
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            print("Goodbye!")
            break
    
        response = conversation_chain.invoke(user_question)
        chat_history = response['chat_history']
        # chat_history: [HumanMessage(content='...'), AIMessage(content='...'), HumanMessage(content='...'), AIMessage(content='...')]
        bot_response = chat_history[-1].content if chat_history else "I'm sorry, I don't understand."
        print("EduAssit:", bot_response)
        question_count += 1
        
        # Process the most relavant images
        tokenized_descriptions = [row[1] for row in rows]
        
        # Compute BERT embeddings
        question_embedding = model.encode([user_question])[0]
        description_embeddings = model.encode(tokenized_descriptions)
        
        # Compute similarity scores
        similarity_scores = []
        for row, description_embedding in zip(rows, description_embeddings):
            # print("question_embedding", question_embedding.reshape(1, -1))
            # print("description_embedding", description_embedding.reshape(1, -1))
            similarity_score = cosine_similarity(question_embedding.reshape(1, -1), description_embedding.reshape(1, -1))[0][0]
            similarity_scores.append((row[0], row[1], similarity_score))
            
        
        # show the most relevant image
        max_row = max(similarity_scores, key=lambda x: x[2])
        if max_row[2] > 0.3:
            print("the most relevant image_path:", max_row[0])
            print("the description of the image:", max_row[1])
        


if __name__ == "__main__":
    main()