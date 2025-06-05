__import__('pysqlite3')
import sys
import os
from dotenv import load_dotenv
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from utils.Transcript import Transcript
from DataBases.VectorStore import VectorStore
from utils.LLM import GeminiLLM
from utils.Summarizer import Summarizer
from utils.HelperFunctions import create_chunks_with_timestamps, get_video_id

transcript = Transcript()


st.title("YouTube QnA")

with st.sidebar:
    
    # all inputs
    video_url = st.text_input("YouTube Video URL", key="video_url")
    groq_api_key = st.text_input("Groq API Key", key="groq_api_key", type="password")
    gemini_api_key = st.text_input("Gemini API Key", key="gemini_api_key", type="password")


    video_process_button = st.button("Process")
    if video_process_button:
        if video_url and groq_api_key and gemini_api_key:
            
            with st.status("Processing data...", expanded=True) as status:
                # get video id
                video_id = get_video_id(video_url)
                if "video_id" not in st.session_state:
                    st.session_state.video_id = video_id

                if not video_id:
                    st.error("Could not extract video ID from URL")
                
                else:
                    st.write("Video ID extracted successfully")

                    if "vector_store" not in st.session_state:
                        st.session_state.vector_store = VectorStore(gemini_api_key)


                    # check if transcript already exist
                    if len(st.session_state.vector_store.collection.get(where={"youtube_id": video_id})['ids']) > 0:
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
                            chunks, timestamps = create_chunks_with_timestamps(transcript_list)
                            st.write("Chunks created successfully")

                            # store in vector db
                            st.session_state.vector_store.add_documents(chunks, timestamps, video_id)

                            if "chunks_for_summarization" not in st.session_state:
                                st.session_state.chunks_for_summarization = chunks
                            st.write("Documents added to vector db successfully")
                status.update(
                    label="Processing complete!", state="complete", expanded=False
                )
        else:
            st.error("All fields are required")
        

if video_url and groq_api_key and gemini_api_key:

    if 'summary' not in st.session_state:
        summarizer = Summarizer(gemini_api_key)
        if ("chunks_for_summarization" not in st.session_state) & ("vector_store" in st.session_state):
            chunks_for_summarization = st.session_state.vector_store.collection.get(
                where={"youtube_id": st.session_state.video_id},
                include=["documents"],
            )['documents']
            st.session_state.chunks_for_summarization = chunks_for_summarization

        if "chunks_for_summarization" in st.session_state:
            with st.spinner("Summarizing..."):
                st.session_state.summary = summarizer.summarize_transcript(st.session_state.chunks_for_summarization)
            st.success(st.session_state.summary)



    gemini_llm = GeminiLLM(gemini_api_key)

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
        realted_chunks = st.session_state.vector_store.retrieve_documents(prompt, st.session_state.video_id)
        system_instruction =f"""
Instructions:
- Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
- Utilize the context provided for accurate and specific information.
- Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
- Cite your sources(here it it "start" time, where the chunk was played). Cites most be in markdown hyperlink format: [start](https://www.youtube.com/watch?v={st.session_state.video_id}&t=start) 
Context:
{realted_chunks}
        """

        # Display assistant response in chat message container
        with st.chat_message("model", avatar="🤖"):
            
            with st.spinner("Thinking..."):
                response = gemini_llm.TextLLM(
                        system_instruction=system_instruction,
                        history=st.session_state.messages,
                        query=prompt,
                    )
            st.markdown(response)
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "model", "parts": [{"text": response}]})
