"""
main.py

This script sets up a Streamlit application for YouTube QnA.
It allows users to input a YouTube video URL, process its transcript,
summarize it, and then ask questions based on the video content.
"""

__import__('pysqlite3')
import sys
import os
import logging
from dotenv import load_dotenv
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from utils.Transcript import Transcript
from DataBases.VectorStore import VectorStore
from utils.LLM import GeminiLLM
from utils.Summarizer import Summarizer
from utils.HelperFunctions import create_chunks_with_timestamps, get_video_id
from config.logging_config import setup_logging

# Set up logging for the application
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Transcript object
transcript = Transcript()

st.title("YouTube QnA")
logger.info("Streamlit application started.")

with st.sidebar:
    logger.info("Sidebar opened for input.")
    # All inputs for video URL and API keys
    video_url = st.text_input("YouTube Video URL", key="video_url")
    groq_api_key = st.text_input("Groq API Key", key="groq_api_key", type="password")
    gemini_api_key = st.text_input("Gemini API Key", key="gemini_api_key", type="password")

    video_process_button = st.button("Process")
    if video_process_button:
        logger.info("Process button clicked.")
        if video_url and groq_api_key and gemini_api_key:
            logger.info("All required fields are filled.")
            with st.status("Processing data...", expanded=True) as status:
                # Get video ID from URL
                video_id = get_video_id(video_url)
                if "video_id" not in st.session_state:
                    st.session_state.video_id = video_id
                logger.debug(f"Extracted video ID: {video_id}")

                if not video_id:
                    st.error("Could not extract video ID from URL")
                    logger.error(f"Failed to extract video ID from URL: {video_url}")
                else:
                    st.write("Video ID extracted successfully")
                    logger.info("Video ID extracted successfully.")

                    if "vector_store" not in st.session_state:
                        st.session_state.vector_store = VectorStore(gemini_api_key)
                        logger.info("VectorStore initialized.")

                    # Check if transcript already exists in vector DB
                    if len(st.session_state.vector_store.collection.get(where={"youtube_id": video_id})['ids']) > 0:
                        st.write("Transcript already exist in vector db")
                        logger.info(f"Transcript for video ID {video_id} already exists in vector DB.")
                    else:
                        st.write("Downloading transcript...")
                        logger.info(f"Attempting to download transcript for video ID: {video_id}")
                        
                        transcript_list = []
                        # Try to get transcript with YouTube API
                        try:
                            transcript_list = transcript.with_youtube_api(video_id)
                            st.write("Transcript successfully with YouTube API")
                            logger.info("Transcript downloaded successfully using YouTube API.")
                        except Exception as e:
                            logger.warning(f"YouTube API failed: {e}. Trying with Whisper.")
                            st.write("Oops YouTube API failed, trying with Whisper")
                            # Fallback to Whisper if YouTube API fails
                            transcript_list = transcript.with_whisper(groq_api_key, video_id)
                            st.write("Transcript successfully with Whisper")
                            logger.info("Transcript downloaded successfully using Whisper.")
                            
                        if not transcript_list:
                            st.error("Transcript failed, Could you try with a different video?")
                            logger.error(f"Transcript download failed for video ID: {video_id}")
                        else:
                            # Create chunks from transcript
                            chunks, timestamps = create_chunks_with_timestamps(transcript_list)
                            st.write("Chunks created successfully")
                            logger.info(f"Created {len(chunks)} chunks from transcript.")

                            # Store chunks in vector DB
                            st.session_state.vector_store.add_documents(chunks, timestamps, video_id)
                            if "chunks_for_summarization" not in st.session_state:
                                st.session_state.chunks_for_summarization = chunks
                            st.write("Documents added to vector db successfully")
                            logger.info("Documents added to vector DB successfully.")
                status.update(
                    label="Processing complete!", state="complete", expanded=False
                )
                logger.info("Video processing complete.")
        else:
            st.error("All fields are required")
            logger.warning("User tried to process without filling all required fields.")
        
# Main application logic after processing
if video_url and groq_api_key and gemini_api_key:
    logger.info("Video URL and API keys are present for main application logic.")

    if 'summary' not in st.session_state:
        logger.info("Summary not found in session state, initiating summarization.")
        summarizer = Summarizer(gemini_api_key)
        if ("chunks_for_summarization" not in st.session_state) and ("vector_store" in st.session_state):
            logger.info("Retrieving chunks for summarization from vector store.")
            chunks_for_summarization = st.session_state.vector_store.collection.get(
                where={"youtube_id": st.session_state.video_id},
                include=["documents"],
            )['documents']
            st.session_state.chunks_for_summarization = chunks_for_summarization
            logger.debug(f"Retrieved {len(chunks_for_summarization)} chunks for summarization.")

        if "chunks_for_summarization" in st.session_state:
            with st.spinner("Summarizing..."):
                logger.info("Starting summarization process.")
                st.session_state.summary = summarizer.summarize_transcript(st.session_state.chunks_for_summarization)
            st.success(st.session_state.summary)
            logger.info("Summarization complete and displayed.")
        else:
            logger.warning("No chunks available for summarization.")

    gemini_llm = GeminiLLM(gemini_api_key)
    logger.info("GeminiLLM initialized for chat.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Chat history initialized.")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["parts"][0]['text'])
    logger.debug("Displayed chat history messages.")

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        logger.info(f"User input received: {prompt}")
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create system instruction for LLM
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
        logger.debug("System instruction for LLM created.")

        # Display assistant response in chat message container
        with st.chat_message("model", avatar="🤖"):
            
            with st.spinner("Thinking..."):
                logger.info("Calling Gemini LLM for response.")
                response = gemini_llm.TextLLM(
                        system_instruction=system_instruction,
                        history=st.session_state.messages,
                        query=prompt,
                    )
            st.markdown(response)
            logger.info("LLM response received and displayed.")
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "model", "parts": [{"text": response}]})
        logger.debug("Chat history updated.")
