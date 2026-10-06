# processor/video_embedder.py

# Imports
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_EMBED_WORKERS = 8

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
# Parallel Video Window Embeddings
################################################################

def embed_video_windows_parallel(
    gcs_uri: str,
    mime_type: str,
    total_duration_sec: int,
) -> dict[int, list[float] | None]:
    """
    Splits the video timeline into 30s windows and embeds 8 windows concurrently
    using gemini-embedding-2. Returns {segment_index: video_embedding}.
    """
    effective_duration = max(total_duration_sec, 1)

    # Build list of (segment_index, start_sec, end_sec)
    windows: list[tuple[int, int, int]] = []
    seg_idx = 0
    start_sec = 0
    while start_sec < effective_duration:
        end_sec = min(start_sec + SEGMENT_INTERVAL_SEC, effective_duration)
        windows.append((seg_idx, start_sec, end_sec))
        seg_idx += 1
        start_sec = end_sec

    total_segments = len(windows)
    video_embeddings: dict[int, list[float] | None] = {}
    completed_count = 0

    def _worker(win: tuple[int, int, int]) -> tuple[int, list[float] | None]:
        idx, s_sec, e_sec = win
        emb = _embed_video_segment(gcs_uri, mime_type, s_sec, e_sec)
        return idx, emb

    with ThreadPoolExecutor(max_workers=MAX_EMBED_WORKERS) as executor:
        futures = {executor.submit(_worker, w): w for w in windows}
        for future in as_completed(futures):
            idx, emb = future.result()
            video_embeddings[idx] = emb
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_segments:
                print(
                    f"  [Video Embed] Completed {completed_count}/{total_segments} video segments...",
                    flush=True,
                )

    return video_embeddings

################################################################
# Build All Multimodal Segments for a Video (Parallel)
################################################################

def build_multimodal_segments_parallel(
    uid: str,
    video_id: str,
    gcs_uri: str,
    mime_type: str,
    total_duration_sec: int,
    words: list[dict[str, Any]],
    precomputed_video_embeddings: dict[int, list[float] | None] | None = None,
) -> list[dict[str, Any]]:
    """
    Combines 30s video embeddings with aligned Speech-to-Text words and
    parallel-embedded text vectors, returning segments ordered by segment_index.
    """
    effective_duration = max(total_duration_sec, 1)

    # 1. Use precomputed video embeddings if provided, otherwise compute in parallel now
    video_embeddings = (
        precomputed_video_embeddings
        if precomputed_video_embeddings is not None
        else embed_video_windows_parallel(gcs_uri, mime_type, effective_duration)
    )

    # 2. Group words into each 30s window
    window_Payloads: list[dict[str, Any]] = []
    segment_index = 0
    start_sec = 0

    while start_sec < effective_duration:
        end_sec = min(start_sec + SEGMENT_INTERVAL_SEC, effective_duration)
        segment_words = [
            w for w in words
            if w["start_sec"] >= float(start_sec) and w["start_sec"] < float(end_sec)
        ]
        transcript_text = " ".join(w["word"] for w in segment_words).strip()

        window_Payloads.append({
            "video_id": video_id,
            "uid": uid,
            "segment_index": segment_index,
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "transcript_text": transcript_text,
            "words": segment_words,
            "video_embedding": video_embeddings.get(segment_index),
        })

        segment_index += 1
        start_sec = end_sec

    # 3. Embed all transcript text snippets in parallel (8 at a time)
    total_segments = len(window_Payloads)
    print(f"  [Text Embed] Embedding {total_segments} text windows in parallel...", flush=True)

    def _text_worker(item: dict[str, Any]) -> dict[str, Any]:
        item["text_embedding"] = _embed_text_snippet(item["transcript_text"])
        return item

    segments: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_EMBED_WORKERS) as executor:
        futures = [executor.submit(_text_worker, item) for item in window_Payloads]
        for future in as_completed(futures):
            segments.append(future.result())

    # Sort back into chronological order (0, 1, 2, ...) since threads finish out of order
    segments.sort(key=lambda s: s["segment_index"])
    return segments