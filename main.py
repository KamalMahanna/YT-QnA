import logging
import os
import sys

# ChromaDB requires sqlite3 >= 3.35.0; replace with pysqlite3 if available
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from dotenv import load_dotenv
import streamlit as st

# Configure application logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yt_qna")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not groq_api_key or not gemini_api_key:
    st.error("API keys missing. Please provide GROQ_API_KEY and GEMINI_API_KEY in your .env file or environment.")
    st.stop()

from DataBases.VectorStore import VectorStore
from utils.HelperFunctions import (
    create_chunks_with_timestamps,
    get_video_id,
    wave_bytesio,
)
from utils.LLM import GeminiLLM
from utils.Summarizer import Summarizer
from utils.Transcript import Transcript

# Streamlit App Setup
st.set_page_config(page_title="YouTube QnA", page_icon="🎬", layout="wide")
st.title("YouTube QnA")

transcript_service = Transcript()
gemini_llm = GeminiLLM(gemini_api_key)

with st.sidebar:
    st.subheader("Video Configuration")
    video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", key="video_url")

    process_col, reset_col = st.columns([1, 1])
    with process_col:
        video_process_button = st.button("Process", type="primary", width="stretch")
    with reset_col:
        delete_collection_button = st.button("Reset DB", width="stretch")

    if delete_collection_button:
        if "vector_store" in st.session_state and st.session_state.vector_store:
            try:
                st.session_state.vector_store.delete_collection()
                st.session_state.vector_store = None
                st.session_state.pop("video_id", None)
                st.session_state.pop("summary", None)
                st.session_state.pop("chunks_for_summarization", None)
                st.session_state.messages = []
                st.success("Database and session reset successfully.")
                logger.info("Database and session state successfully reset by user")
            except Exception as e:
                logger.exception("Error resetting database: %s", e)
                st.error(f"Error resetting database: {e}")
        else:
            st.info("No active vector store to delete.")

    if video_process_button:
        if not video_url.strip():
            st.error("Please enter a valid YouTube URL.")
        else:
            video_id = get_video_id(video_url)
            if not video_id:
                st.error("Could not extract a valid video ID from URL.")
                logger.warning("Failed to extract video ID from URL: %s", video_url)
            else:
                st.session_state.video_id = video_id
                logger.info("Processing video ID: %s", video_id)

                with st.status("Processing YouTube video...", expanded=True) as status:
                    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
                        st.session_state.vector_store = VectorStore(gemini_api_key)

                    # Check if video was previously indexed
                    existing = st.session_state.vector_store.collection.get(
                        where={"youtube_id": video_id},
                        limit=1,
                    )

                    if existing and existing.get("ids"):
                        st.write("Video transcript is already indexed in the vector database.")
                        logger.info("Video %s already indexed in ChromaDB", video_id)
                    else:
                        st.write("Fetching video transcript...")
                        transcript_list = []
                        try:
                            transcript_list = transcript_service.with_youtube_api(video_id)
                            st.write("Retrieved transcript via YouTube Subtitles API.")
                        except Exception as yt_err:
                            logger.warning("YouTube API captions failed for %s: %s; falling back to Whisper STT", video_id, yt_err)
                            st.write("YouTube captions unavailable. Downloading audio and transcribing with Whisper...")
                            try:
                                transcript_list = transcript_service.with_whisper(groq_api_key, video_id)
                                st.write("Transcribed successfully with Whisper STT.")
                            except Exception as whisper_err:
                                logger.exception("Whisper transcription failed for %s: %s", video_id, whisper_err)
                                st.error(f"Transcription failed: {whisper_err}")

                        if not transcript_list:
                            st.error("Could not obtain transcript for this video. Please try another video.")
                            logger.error("Empty transcript list for video ID: %s", video_id)
                        else:
                            st.write("Chunking transcript and aligning timestamps...")
                            chunks, timestamps = create_chunks_with_timestamps(transcript_list)
                            st.write(f"Generated {len(chunks)} searchable chunks.")

                            st.write("Generating embeddings and indexing in vector database...")
                            st.session_state.vector_store.add_documents(chunks, timestamps, video_id)
                            st.write("Indexed documents successfully!")
                            logger.info("Successfully indexed %d chunks for video %s", len(chunks), video_id)

                    status.update(label="Processing complete!", state="complete", expanded=False)

# Main Area: Summary & Conversational Q&A
if video_url and "video_id" in st.session_state and "vector_store" in st.session_state and st.session_state.vector_store:
    current_video_id = st.session_state.video_id

    # Transcript Summarization
    if "summary" not in st.session_state:
        summarizer = Summarizer(gemini_api_key)
        if "chunks_for_summarization" not in st.session_state:
            docs_data = st.session_state.vector_store.collection.get(
                where={"youtube_id": current_video_id},
                include=["documents"],
            )
            st.session_state.chunks_for_summarization = docs_data.get("documents", [])

        if st.session_state.chunks_for_summarization:
            with st.spinner("Generating video summary..."):
                try:
                    summary_text = summarizer.summarize_transcript(
                        st.session_state.chunks_for_summarization
                    )
                    st.session_state.summary = summary_text
                except Exception as sum_err:
                    logger.exception("Summarization failed for %s: %s", current_video_id, sum_err)
                    st.warning("Could not generate summary.")

    if st.session_state.get("summary"):
        with st.container(border=True):
            st.subheader("Video Summary")
            st.markdown(st.session_state.summary)

            if st.button("Listen to Audio Summary", key="tts_btn"):
                with st.spinner("Synthesizing audio..."):
                    try:
                        pcm_audio = gemini_llm.TTS(st.session_state.summary)
                        st.audio(wave_bytesio(pcm_audio), format="audio/wav")
                        logger.info("TTS audio rendered successfully")
                    except Exception as tts_err:
                        logger.warning("TTS audio generation failed: %s", tts_err)
                        st.warning("Audio synthesis unavailable or rate limited.")

    st.divider()

    # Chat History & Q&A
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["parts"][0]["text"])

    if prompt := st.chat_input("Ask a question about the video..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})

        # Retrieve relevant context
        related_chunks = st.session_state.vector_store.retrieve_documents(
            query=prompt,
            video_id=current_video_id,
            n_results=5,
        )

        system_instruction = f"""
Instructions:
- Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'.
- Utilize the context provided for accurate and specific information.
- Incorporate preexisting knowledge only to enhance clarity, but keep claims grounded in the context.
- Cite your sources using markdown hyperlinks pointing to the video timestamp format: [MM:SS](https://www.youtube.com/watch?v={current_video_id}&t=start_seconds) where start_seconds is the integer start time.

Context:
{related_chunks}
"""

        with st.chat_message("model", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    # Pass previous conversation turns without current turn (which send_message provides)
                    history_context = st.session_state.messages[:-1]
                    response_text = gemini_llm.TextLLM(
                        system_instruction=system_instruction,
                        history=history_context,
                        query=prompt,
                    )
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "model", "parts": [{"text": response_text}]})
                    logger.info("Answer generated successfully for query '%s'", prompt[:50])
                except Exception as llm_err:
                    logger.exception("Error during LLM response generation: %s", llm_err)
                    st.error("Failed to generate response. Please try again.")
