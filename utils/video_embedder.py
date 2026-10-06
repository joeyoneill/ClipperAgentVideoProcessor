# processor/video_embedder.py

# Imports
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Any
import os

# Load ENV Vars
load_dotenv()

# Constants
EMBEDDING_MODEL = "gemini-embedding-2"
SEGMENT_INTERVAL_SEC = 30

# MAC ONLY
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

# Set up client
client = genai.Client(
    vertexai=True,
    project=os.environ['GCP_PROJECT_ID'],
    location='global'
)

################################################################
# Embeds Single Video Segment
################################################################

def _embed_video_segment(
    gcs_uri: str,
    mime_type: str,
    start_sec: int,
    end_sec: int,
) -> list[float] | None:
    """
    Uses multimedia embedding API with VideoMetadata(start_offset, end_offset)
    to embed a specific time slice of the GCS video without slicing the file.
    """
    # prepare request
    part = types.Part(
        file_data=types.FileData(
            file_uri=gcs_uri,
            mime_type=mime_type,
        ),
        video_metadata=types.VideoMetadata(
            start_offset=f"{start_sec}s",
            end_offset=f"{end_sec}s",
        ),
    )
    content = types.Content(parts=[part])

    # send api request
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[content],
        config=types.EmbedContentConfig(output_dimensionality=1408),
    )

    # return response embeddings
    if not response.embeddings:
        return None
    first_embedding = response.embeddings[0]
    return list(first_embedding.values) if first_embedding.values else None

################################################################
# Embed the Transcript Text for that Window
################################################################
def _embed_text_snippet(text: str) -> list[float] | None:
    """
    Embeds the spoken transcript for a time window using gemini-embedding-2.
    """
    cleaned = text.strip()
    if not cleaned:
        return None
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[cleaned],
        config=types.EmbedContentConfig(output_dimensionality=1408),
    )
    if not response.embeddings:
        return None
    first_embedding = response.embeddings[0]
    return list(first_embedding.values) if first_embedding.values else None

################################################################
# Step C: Build All Multimodal Segments for a Video
################################################################

def build_multimodal_segments(
    uid: str,
    video_id: str,
    gcs_uri: str,
    mime_type: str,
    total_duration_sec: int,
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Steps through the video in 30-second virtual windows, pairing each window's
    gemini-embedding-2 video embedding with its aligned Speech-to-Text words
    and text embedding.
    """
    effective_duration = max(total_duration_sec, 1)
    segments: list[dict[str, Any]] = []

    segment_index = 0
    start_sec = 0
    total_segments = (effective_duration + SEGMENT_INTERVAL_SEC - 1) // SEGMENT_INTERVAL_SEC

    while start_sec < effective_duration:
        end_sec = min(start_sec + SEGMENT_INTERVAL_SEC, effective_duration)
        print(
            f"  [2/4] Embedding segment {segment_index + 1}/{total_segments} "
            f"({start_sec}s - {end_sec}s) with {EMBEDDING_MODEL}...",
            flush=True,
        )

        # 1. Embed the video frames for [start_sec, end_sec]
        video_embedding = _embed_video_segment(
            gcs_uri=gcs_uri,
            mime_type=mime_type,
            start_sec=start_sec,
            end_sec=end_sec,
        )

        # 2. Grab all words from Speech-to-Text V2 spoken in [start_sec, end_sec)
        segment_words = [
            w for w in words
            if w["start_sec"] >= float(start_sec) and w["start_sec"] < float(end_sec)
        ]
        transcript_text = " ".join(w["word"] for w in segment_words).strip()

        # 3. Embed the spoken transcript text in the same gemini-embedding-2 space
        text_embedding = _embed_text_snippet(transcript_text)

        segments.append({
            "video_id": video_id,
            "uid": uid,
            "segment_index": segment_index,
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "transcript_text": transcript_text,
            "words": segment_words,
            "video_embedding": video_embedding,
            "text_embedding": text_embedding,
        })

        segment_index += 1
        start_sec = end_sec

    return segments