# processor/audio_transcriber.py

# Imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import imageio_ffmpeg
import os
import subprocess
import tempfile
import time
from typing import Any

# Custom Dependencies
from utils.storage import gcs_client

# Load Env Vars
load_dotenv()
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCS_ROOT_BUCKET_NAME = os.environ["GCS_ROOT_BUCKET_NAME"]

# Initialize Client
speech_client = SpeechClient()

# Constants
AUDIO_CHUNK_SEC = 600  # 10 minutes per chunk (600 seconds)
MAX_STT_WORKERS = 6    # Max concurrent Speech-to-Text V2 jobs

################################################################
# Extract & Split 16kHz Mono FLAC Chunks via FFmpeg
################################################################

def _extract_and_upload_audio_chunks(
    bucket_name: str,
    video_blob_path: str,
    audio_folder_prefix: str,
) -> list[tuple[int, float, str, str]]:
    """
    1. Downloads the uploaded video from GCS to a temporary disk file.
    2. Uses ffmpeg segment muxer to extract 16kHz mono FLAC split into 10-min chunks
       and records the exact start timestamp of each chunk in segments.csv.
    3. Uploads all chunk_XXX.flac files to GCS in parallel.
    4. Returns list of (chunk_index, exact_start_sec, blob_path, gcs_uri).
    """
    bucket = gcs_client.bucket(bucket_name)
    video_blob = bucket.blob(video_blob_path)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_video_path = os.path.join(tmp_dir, "input_video")
        chunk_pattern = os.path.join(tmp_dir, "chunk_%03d.flac")
        segment_csv_path = os.path.join(tmp_dir, "segments.csv")
        
        # 1. Download video from GCS
        print(f"  [1a] Downloading {video_blob_path} from GCS...", flush=True)
        video_blob.download_to_filename(local_video_path)
        print(f"  [1a] Downloaded {video_blob_path} from GCS.", flush=True)
        
        # 2. Extract 16kHz mono FLAC and record exact chunk start times in segments.csv
        print(
            f"  [1b] Extracting & splitting 16kHz mono FLAC into {AUDIO_CHUNK_SEC}s chunks...",
            flush=True,
        )
        ffmpeg_cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i", local_video_path,
            "-vn",                              # Drop video stream (audio only)
            "-ac", "1",                         # Mono channel
            "-ar", "16000",                     # 16kHz sample rate
            "-f", "segment",                    # Split into chunks
            "-segment_time", str(AUDIO_CHUNK_SEC),
            "-reset_timestamps", "1",           # Each chunk starts at 0.0s
            "-segment_list", segment_csv_path,  # Write exact start/end times to CSV
            "-segment_list_type", "csv",
            chunk_pattern,
        ]
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        
        # Parse exact (chunk_index, filename, exact_start_sec) from ffmpeg's segments.csv
        chunk_meta: list[tuple[int, str, float]] = []
        with open(segment_csv_path, "r", encoding="utf-8") as csv_file:
            for idx, line in enumerate(csv_file):
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    fname = parts[0]
                    exact_start_sec = float(parts[1])
                    chunk_meta.append((idx, fname, exact_start_sec))
        
        print(f"  [1b] Created {len(chunk_meta)} audio chunk(s).", flush=True)
        
        # 3. Upload all chunks to GCS in parallel
        print(f"  [1c] Uploading {len(chunk_meta)} audio chunk(s) to GCS...", flush=True)
        
        def _upload_one(item: tuple[int, str, float]) -> tuple[int, float, str, str]:
            idx, filename, exact_start_sec = item
            local_chunk_path = os.path.join(tmp_dir, filename)
            blob_path = f"{audio_folder_prefix}/{filename}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_chunk_path, content_type="audio/flac")
            return idx, exact_start_sec, blob_path, f"gs://{bucket_name}/{blob_path}"
        
        uploaded_chunks: list[tuple[int, float, str, str]] = []
        with ThreadPoolExecutor(max_workers=MAX_STT_WORKERS) as executor:
            futures = [executor.submit(_upload_one, item) for item in chunk_meta]
            for future in as_completed(futures):
                uploaded_chunks.append(future.result())
        
        uploaded_chunks.sort(key=lambda x: x[0])
        return uploaded_chunks

################################################################
# Transcribe a Single 10-Minute Audio Chunk
################################################################

def _transcribe_single_chunk(
    chunk_index: int,
    total_chunks: int,
    chunk_start_sec: float,
    audio_gcs_uri: str,
) -> tuple[int, str, list[dict[str, Any]]]:
    """
    Runs Speech-to-Text V2 BatchRecognize on one 10-minute audio chunk and shifts
    all word timestamps by the exact chunk_start_sec from ffmpeg.
    """
    recognizer_path = f"projects/{GCP_PROJECT_ID}/locations/global/recognizers/_"

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="long",
        features=cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        ),
    )

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=recognizer_path,
        config=config,
        files=[cloud_speech.BatchRecognizeFileMetadata(uri=audio_gcs_uri)],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig()
        ),
    )

    operation = speech_client.batch_recognize(request=request)

    start_wait = time.time()
    while not operation.done():
        elapsed = int(time.time() - start_wait)
        print(
            f"  [1d] Chunk {chunk_index + 1}/{total_chunks} STT progress: "
            f"Queued/Running (elapsed: {elapsed}s)",
            flush=True,
        )
        time.sleep(15)

    response = operation.result(timeout=60)
    if response is None:
        raise RuntimeError(f"Speech-to-Text V2 returned empty response for chunk {chunk_index}.")

    file_result = response.results[audio_gcs_uri]
    if file_result.error and file_result.error.code != 0:
        raise RuntimeError(
            f"Speech-to-Text V2 error on chunk {chunk_index}: {file_result.error.message}"
        )

    transcript_parts: list[str] = []
    words: list[dict[str, Any]] = []

    transcript_obj = file_result.inline_result.transcript or file_result.transcript
    for result in transcript_obj.results:
        if not result.alternatives:
            continue
        best_alt = result.alternatives[0]
        if best_alt.transcript:
            transcript_parts.append(best_alt.transcript.strip())

        for word_info in best_alt.words:
            # Read exact protobuf seconds + nanoseconds and add exact ffmpeg chunk_start_sec
            raw_start = word_info._pb.start_offset
            raw_end = word_info._pb.end_offset
            start_sec = raw_start.seconds + (raw_start.nanos / 1_000_000_000.0) + chunk_start_sec
            end_sec = raw_end.seconds + (raw_end.nanos / 1_000_000_000.0) + chunk_start_sec

            words.append({
                "word": word_info.word,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
            })

    chunk_transcript = " ".join(transcript_parts)
    print(
        f"  [1d] Chunk {chunk_index + 1}/{total_chunks} complete! ({len(words)} words)",
        flush=True,
    )
    return chunk_index, chunk_transcript, words

################################################################
# Public Function (Extract Chunks -> Parallel STT -> Cleanup)
################################################################

def transcribe_video_from_gcs(
    uid: str,
    video_id: str,
    video_blob_path: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Splits audio into 10-minute chunks, transcribes all chunks in parallel via
    Speech-to-Text V2, merges transcripts & offset word timestamps in order,
    and deletes all temporary chunk blobs from GCS.
    """
    audio_folder_prefix = f"{uid}/lf_videos/{video_id}/temp_audio_chunks"
    bucket = gcs_client.bucket(GCS_ROOT_BUCKET_NAME)
    uploaded_chunks: list[tuple[int, float, str, str]] = []

    try:
        uploaded_chunks = _extract_and_upload_audio_chunks(
            bucket_name=GCS_ROOT_BUCKET_NAME,
            video_blob_path=video_blob_path,
            audio_folder_prefix=audio_folder_prefix,
        )
        total_chunks = len(uploaded_chunks)
        print(
            f"  [1d] Starting {total_chunks} parallel Speech-to-Text V2 job(s)...",
            flush=True,
        )

        # Transcribe all 10-minute chunks simultaneously
        chunk_results: list[tuple[int, str, list[dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=MAX_STT_WORKERS) as executor:
            futures = [
                executor.submit(
                    _transcribe_single_chunk,
                    idx,
                    total_chunks,
                    chunk_start_sec,
                    gcs_uri,
                )
                for idx, chunk_start_sec, _, gcs_uri in uploaded_chunks
            ]
            for future in as_completed(futures):
                chunk_results.append(future.result())

        # Sort chunks back into chronological order (0, 1, 2, ...)
        chunk_results.sort(key=lambda r: r[0])

        full_transcript = " ".join(r[1] for r in chunk_results if r[1]).strip()
        all_words: list[dict[str, Any]] = []
        for _, _, chunk_words in chunk_results:
            all_words.extend(chunk_words)

        print(
            f"  [1d] All {total_chunks} chunk(s) transcribed! Total: {len(all_words)} words.",
            flush=True,
        )
        return full_transcript, all_words

    finally:
        # Always clean up all temporary audio chunk blobs from GCS
        for _, _, blob_path, _ in uploaded_chunks:
            temp_blob = bucket.blob(blob_path)
            if temp_blob.exists():
                temp_blob.delete()