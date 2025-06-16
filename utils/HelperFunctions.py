from nltk.tokenize import sent_tokenize
import re
import io
import wave


def create_chunks_with_timestamps(transcript_list):
    chunk_size = 500
    chunks = []
    timestamps = []

    chunk_stack = ""
    timestamp_stack = ""

    for i in transcript_list:
        the_text = i["text"].strip()
        the_timestamp = i["start"]

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
                while len(chunk_stack := ". ".join(splits)) > chunk_size:

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


def chunk_by_sentences(text, max_chars=100_000):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def get_video_id(url):
    video_ids = re.findall(r"(?:v=|\/)([\w-]{11}).*", url)

    if video_ids:
        return video_ids[0]
    else:
        return ""


def wave_bytesio(pcm: bytes, channels=1, rate=24000, sample_width=2) -> io.BytesIO:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    buffer.seek(0)
    return buffer
