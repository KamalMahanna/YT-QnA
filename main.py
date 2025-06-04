import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

import streamlit as st
from utils.UrlHelper import UrlHelper
from utils.Transcript import Transcript
# from utils.Translator import translation_needed, bulk_translate
from DataBases.VectorStore import VectorStore
from utils.LLM import GeminiLLM

url_helper = UrlHelper()
transcript = Transcript()

def create_chunks(transcript_list):
    chunk_size = 500
    chunks = []
    timestamps = []

    chunk_stack = ""
    timestamp_stack = ""

    for i in transcript_list:
        the_text = i['text'].strip()
        the_timestamp = i['start']
            
        # if the text is longer than the chunk size
        if len(the_text) >= chunk_size:

            # if previously any chunks exist, append them to the list
            if chunk_stack:
                chunks.append(chunk_stack)
                timestamps.append(timestamp_stack)

                # reset the stack
                chunk_stack = ""
                timestamp_stack = 0

            # append the current chunk
            chunks.append(the_text)
            timestamps.append(the_timestamp)


        # if the text is shorter than the chunk size
        else:
            
            # if chunk and text combined is longer than the chunk size
            if len(chunk_stack := chunk_stack + " " + the_text) > chunk_size:
                
                # split the chunk stack into sentences
                splits = chunk_stack.split(". ")
                temp_chunk_stack = ""
                
                # while the chunk stack is longer than the chunk size
                while(len(chunk_stack := ". ".join(splits)) > chunk_size):
                    
                    # pop the last sentence from the splits and add it to the temp chunk stack
                    temp_chunk_stack = splits.pop() + ". " + temp_chunk_stack
                
                # append the chunk stack and timestamp when the chunk stack is shorter than the chunk size
                chunks.append(chunk_stack.strip())
                timestamps.append(timestamp_stack)

                # reset the stack to temp chunk stack
                chunk_stack = temp_chunk_stack

            # always update the timestamp    
            timestamp_stack = the_timestamp

    return chunks, timestamps



st.title("YouTube QnA")

with st.sidebar:
    
    video_url = st.text_input("YouTube Video URL", key="video_url")

    groq_api_key = st.text_input("Groq API Key", key="groq_api_key", type="password")
    gemini_api_key = st.text_input("Gemini API Key", key="gemini_api_key", type="password")

    video_process_button = st.button("Process",)
    if video_process_button:
        if video_url and groq_api_key and gemini_api_key:
            
            with st.status("Processing data...", expanded=True) as status:
                # get video id
                video_id = url_helper.get_video_id(video_url)
                if "video_id" not in st.session_state:
                    st.session_state.video_id = video_id

                if not video_id:
                    st.error("Could not extract video ID from URL")
                
                else:
                    st.write("Video ID extracted successfully")

                    vector_store = VectorStore(gemini_api_key)

                    # check if transcript already exist
                    if len(vector_store.collection.get(where={"youtube_id": video_id})['ids']) > 0:
                        st.write("Transcript already exist in vector db")
                    
                    else:
                        st.write("Downloading transcript...")
                        
                        transcript_list = []
                        # transcript with youtube api
                        try:
                            transcript_list = transcript.with_youtube_api(video_id)
                            st.write("Transcript successfully with YouTube API")
                        except:
                            st.write("Oops YouTube API failed, trying with Whisper")
                            transcript_list = transcript.with_whisper(groq_api_key, video_id)
                            st.write("Transcript successfully with Whisper")
                            
                        if not transcript_list:
                            st.error("Transcript failed, Could you try with a different video?")
                        else:
                            # create chunks
                            chunks, timestamps = create_chunks(transcript_list)
                            st.write("Chunks created successfully")

                            # translate
                            # if translation_needed(chunks):
                            #     transcript_list = bulk_translate(chunks)
                            #     st.write("Transcript translated successfully")
                            
                            # else:
                            #     st.write("Transcript already in English")
                            


                            # store in vector db
                            vector_store.add_documents(chunks, timestamps, video_id)
                            st.write("Documents added to vector db successfully")
                status.update(
                    label="Processing complete!", state="complete", expanded=False
                )
        else:
            st.error("All fields are required")
        

if video_url and groq_api_key and gemini_api_key:
    gemini_llm = GeminiLLM(gemini_api_key)
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(gemini_api_key)
    vector_store = st.session_state.vector_store

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["parts"][0]['text'])

    # Accept user input
    if prompt := st.chat_input("What is up?"):

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # create system_instruction
        realted_chunks = vector_store.retrieve_documents(prompt, st.session_state.video_id)
        system_instruction =f"""
Instructions:
- Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
- Utilize the context provided for accurate and specific information.
- Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
- Cite your sources(here it it start)
Context:
{realted_chunks}
            """

        # Display assistant response in chat message container
        with st.chat_message("model", avatar="🤖"):
            response = st.write_stream(
                gemini_llm.TextLLM(
                    system_instruction=system_instruction,
                    history=st.session_state.messages,
                    query=prompt,
                )
            )
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "model", "parts": [{"text": response}]})
