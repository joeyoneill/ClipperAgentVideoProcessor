# processor/audio_transcriber.py

# Imports
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import os
import subprocess
import tempfile
from typing import Any

# Load Env Vars
load_dotenv()
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCS_ROOT_BUCKET_NAME = os.environ["GCS_ROOT_BUCKET_NAME"]

# Initialize Clients
gcs_client = storage.Client()
speech_client = SpeechClient()


################################################################
# Step A: Extract 16kHz Mono FLAC Audio via FFmpeg
################################################################

def _extract_and_upload_temp_audio(
    bucket_name: str,
    video_blob_path: str,
    audio_blob_path: str
) -> str:
    """
    1. Downloads the uploaded video from GCS to a temporary disk file.
    2. Uses ffmpeg to strip video frames (-vn) and convert audio to 16kHz mono FLAC.
    3. Uploads temp_audio.flac to GCS for Speech-to-Text V2 BatchRecognize.
    4. Cleans up local temp files immediately.
    """
    bucket = gcs_client.bucket(bucket_name)
    video_blob = bucket.blob(video_blob_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_video_path = os.path.join(tmp_dir, "input_video")
        local_audio_path = os.path.join(tmp_dir, "temp_audio.flac")

        # 1. Download video from GCS
        video_blob.download_to_filename(local_video_path)

        # 2. Extract lightweight mono 16kHz FLAC audio via ffmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",                   # Overwrite output if exists
            "-i", local_video_path, # Input video file
            "-vn",                  # Drop video stream (audio only)
            "-ac", "1",             # Mono channel (ideal for speech recognition)
            "-ar", "16000",         # 16kHz sample rate
            local_audio_path,
        ]
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # 3. Upload extracted audio to GCS
        audio_blob = bucket.blob(audio_blob_path)
        audio_blob.upload_from_filename(local_audio_path, content_type="audio/flac")

    return f"gs://{bucket_name}/{audio_blob_path}"

################################################################
# Step B: Run Speech-to-Text V2 BatchRecognize
################################################################

def _run_batch_recognize(audio_gcs_uri: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Calls Google Cloud Speech-to-Text V2 BatchRecognize on the GCS audio URI
    and returns (full_transcript, list_of_word_timestamps).
    """
    recognizer_path = f"projects/{GCP_PROJECT_ID}/locations/global/recognizers/_"

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="long",  # Optimized for long-form video/podcast audio
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

    # Long-running operation: waits for Speech-to-Text V2 to finish
    operation = speech_client.batch_recognize(request=request)
    response = operation.result(timeout=3600)
    if response is None:
        raise RuntimeError("Speech-to-Text V2 returned an empty response.")

    file_result = response.results[audio_gcs_uri]
    if file_result.error and file_result.error.code != 0:
        raise RuntimeError(f"Speech-to-Text V2 error: {file_result.error.message}")

    transcript_parts: list[str] = []
    words: list[dict[str, Any]] = []

    for result in file_result.transcript.results:
        if not result.alternatives:
            continue
        best_alt = result.alternatives[0]
        if best_alt.transcript:
            transcript_parts.append(best_alt.transcript.strip())

        for word_info in best_alt.words:
            start_sec = word_info.start_offset.total_seconds()
            end_sec = word_info.end_offset.total_seconds()
            words.append({
                "word": word_info.word,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
            })

    full_transcript = " ".join(transcript_parts)
    return full_transcript, words

################################################################
# Public Function (Extract -> Transcribe -> Delete Temp Audio)
################################################################

def transcribe_video_from_gcs(
    uid: str,
    video_id: str,
    video_blob_path: str
) -> tuple[str, list[dict[str, Any]]]:
    """
    Extracts audio from the GCS video, runs Speech-to-Text V2 for word timestamps,
    and deletes the temporary audio blob from GCS in a finally block.
    """
    temp_audio_blob_path = f"{uid}/lf_videos/{video_id}/temp_audio.flac"
    bucket = gcs_client.bucket(GCS_ROOT_BUCKET_NAME)

    try:
        audio_gcs_uri = _extract_and_upload_temp_audio(
            bucket_name=GCS_ROOT_BUCKET_NAME,
            video_blob_path=video_blob_path,
            audio_blob_path=temp_audio_blob_path,
        )
        return _run_batch_recognize(audio_gcs_uri)
    finally:
        # Always remove temp_audio.flac from GCS so only original.<ext> remains
        temp_blob = bucket.blob(temp_audio_blob_path)
        if temp_blob.exists():
            temp_blob.delete()