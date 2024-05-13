import sqlite3
import pickle
from youtube_transcript_api import YouTubeTranscriptApi
import tensorflow_hub as hub
import os
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# Replace 'YOUR_API_KEY' with your actual YouTube Data API key
def chatWithAI(transcriptFrom,question):

  print(transcriptFrom)

  print(question)
  
  api_key = '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'
  client = OpenAI(api_key=api_key)
  transcriptFrom = transcriptFrom
  question = question
  completion = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
      {"role": "system", "content": "You are in the capacity of a teaching assistant and I will ask you some questions about the video. You'll answer me based on what I've given you and with a timeline"},
      {"role": "user", "content": "Here i give you a transcript from a video."+transcriptFrom+"And my question is:"+question+".If you quote a line from the video in your answer, you need to specify its time point in the form of a number (in the text of the video, I used start to indicate when it starts)."}
    ]
  )
  print(completion.choices[0].message.content)

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


def get_video_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # transcript_text = '\n'.join([t['text'] for t in transcript_list])
        transcript_text = []
        transcript = ""
        for t in transcript_list:
            transcript = transcript+t['text']+"--start:"+str(t['start'])+"--duration:"+str(t['duration'])+"\n"
            transcript_text.append(t['text']+"--start:"+str(t['start'])+"--duration:"+str(t['duration']))
        print("video to text success")
        print(transcript)
        return transcript_text,transcript
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None,None


def convert_text_to_embeddings(text):
    # Load the Universal Sentence Encoder model
    embed = hub.load("./model")
    texts = text
    # Generate embeddings for the transcript tokens
    embeddings = embed(texts)
    print("embedding success")
    return embeddings

while True:
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Options:")
    print("3. Search a video")
    print("4. Discuss video with AI")
    choice = input("Enter your choice (1/2/3/4): ")
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")

    if choice == '3':
        query = input('input your key words:')
        video_info = search_youtube_videos(query)
        if video_info:
            print('Video ID:', video_info['videoId'])
            print('Title:', video_info['title'])
            print('Description:', video_info['description'])
            print('Url:',video_info['url'])
            print('Date',video_info['date'])
        else:
            print('No video found for the query.')
    elif choice == '4':
            videoId = input("input your video id:")
            transcriptList,transcriptString = get_video_transcript(videoId)
            while True:
                question = input("input your question(enter c to exit):")
                if question=='c':
                    break
                else:
                    chatWithAI(transcriptString,question)
    else:
        print("Invalid choice. Please enter 1 or 2.")