import logging
import re
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')
# Set up logger for this module
logger = logging.getLogger(__name__)


def create_chunks_with_timestamps(transcript_list: list) -> tuple[list, list]:
    """
    Creates text chunks from a transcript list, ensuring each chunk is within
    a specified size and retaining associated timestamps.

    Args:
        transcript_list (list): A list of dictionaries, where each dictionary
                                contains 'text' and 'start' (timestamp).

    Returns:
        tuple[list, list]: A tuple containing two lists:
                           - chunks (list of str): The processed text chunks.
                           - timestamps (list of int): The start timestamps for each chunk.
    """
    logger.info("Starting to create chunks with timestamps.")
    chunk_size = 500
    chunks = []
    timestamps = []

    chunk_stack = ""
    timestamp_stack = 0

    for i, item in enumerate(transcript_list):
        the_text = item['text'].strip()
        the_timestamp = item['start']
        logger.debug(f"Processing transcript item {i}: text='{the_text[:50]}...', timestamp={the_timestamp}")
            
        # If the text is longer than the chunk size, append it directly
        if len(the_text) >= chunk_size:
            if chunk_stack:
                chunks.append(chunk_stack.strip())
                timestamps.append(timestamp_stack)
                logger.debug(f"Appended previous chunk_stack (len: {len(chunk_stack)}).")
                # Reset the stack
                chunk_stack = ""
                timestamp_stack = 0

            chunks.append(the_text)
            timestamps.append(the_timestamp)
            logger.debug(f"Appended large text chunk directly (len: {len(the_text)}).")

        # If the text is shorter than the chunk size
        else:
            # Attempt to add current text to chunk_stack
            proposed_chunk_stack = chunk_stack + " " + the_text if chunk_stack else the_text
            
            # If chunk and text combined is longer than the chunk size
            if len(proposed_chunk_stack) > chunk_size:
                logger.debug(f"Combined chunk_stack exceeds size, splitting sentences. Current stack len: {len(proposed_chunk_stack)}")
                # Split the chunk stack into sentences
                splits = sent_tokenize(proposed_chunk_stack)
                temp_chunk_stack = ""
                
                # While the chunk stack is longer than the chunk size, pop sentences
                while len(". ".join(splits)) > chunk_size and len(splits) > 1:
                    temp_chunk_stack = splits.pop() + ". " + temp_chunk_stack
                
                # Append the current valid chunk stack and its timestamp
                chunks.append(". ".join(splits).strip())
                timestamps.append(timestamp_stack)
                logger.debug(f"Appended chunk after splitting (len: {len(chunks[-1])}).")

                # Reset the stack to the remaining sentences
                chunk_stack = temp_chunk_stack.strip()
                logger.debug(f"Remaining chunk_stack after split: '{chunk_stack[:50]}...'")
            else:
                chunk_stack = proposed_chunk_stack
                logger.debug(f"Added text to chunk_stack. Current stack len: {len(chunk_stack)}.")

            # Always update the timestamp to the start of the current text being processed
            timestamp_stack = the_timestamp
    
    # Append any remaining content in the chunk_stack after the loop
    if chunk_stack:
        chunks.append(chunk_stack.strip())
        timestamps.append(timestamp_stack)
        logger.debug(f"Appended final remaining chunk (len: {len(chunk_stack)}).")

    logger.info(f"Finished creating chunks. Total chunks: {len(chunks)}")
    return chunks, timestamps


def chunk_by_sentences(text: str, max_chars: int = 100_000) -> list:
    """
    Splits a given text into chunks based on sentence boundaries,
    ensuring no chunk exceeds max_chars.

    Args:
        text (str): The input text to be chunked.
        max_chars (int, optional): The maximum character limit for each chunk.
                                   Defaults to 100,000.

    Returns:
        list: A list of text chunks (strings).
    """
    logger.info(f"Starting to chunk text by sentences with max_chars={max_chars}.")
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    for i, sentence in enumerate(sentences):
        # Check if adding the current sentence exceeds the max_chars
        # Add 1 for the space character
        if len(current_chunk) + len(sentence) + (1 if current_chunk else 0) <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
            logger.debug(f"Added sentence {i} to current chunk. Current chunk len: {len(current_chunk)}")
        else:
            # If it exceeds, append the current chunk and start a new one
            chunks.append(current_chunk.strip())
            logger.debug(f"Chunk completed (len: {len(chunks[-1])}). Starting new chunk with sentence {i}.")
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
        logger.debug(f"Appended final chunk (len: {len(current_chunk)}).")
    logger.info(f"Finished chunking by sentences. Total chunks: {len(chunks)}")
    return chunks


def get_video_id(url: str) -> str:
    """
    Extracts the YouTube video ID from a given YouTube URL.

    Args:
        url (str): The YouTube video URL.

    Returns:
        str: The 11-character YouTube video ID, or an empty string if not found.
    """
    logger.info(f"Attempting to extract video ID from URL: {url}")
    video_ids = re.findall(r"(?:v=|\/)([\w-]{11}).*", url)

    if video_ids:
        logger.info(f"Successfully extracted video ID: {video_ids[0]}")
        return video_ids[0]
    else:
        logger.warning(f"Could not extract video ID from URL: {url}")
        return ""
