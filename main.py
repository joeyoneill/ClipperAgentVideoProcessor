# processor/main.py

# Imports
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
import json
import os
from typing import Any

# Local Processor Modules
from utils.db import db
from utils.audio_transcriber import transcribe_video_from_gcs
from utils.storage import gcs_client
from utils.video_embedder import build_multimodal_segments

# Load Env Vars
load_dotenv()
GCS_ROOT_BUCKET_NAME = os.environ["GCS_ROOT_BUCKET_NAME"]
LFVIDEO_COLLECTION_NAME = os.environ["LFVIDEO_COLLECTION_NAME"]
LFVIDEO_SEGMENT_COLLECTION_NAME = os.environ['LFVIDEO_SEGMENT_COLLECTION_NAME']

################################################################
# Helper Functions
################################################################


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def save_segments_to_firestore(
    video_id: str,
    segments: list[dict[str, Any]]
) -> None:
    """
    Saves all 30s virtual segments to Firestore in batches of 400.
    Wraps float lists in Firestore's Vector() type for KNN retrieval.
    """
    collection_ref = db.collection(LFVIDEO_SEGMENT_COLLECTION_NAME)
    batch = db.batch()
    op_count = 0

    for seg in segments:
        seg_idx: int = seg["segment_index"]
        doc_id = f"{video_id}_seg_{seg_idx:04d}"
        doc_ref = collection_ref.document(doc_id)

        # Convert raw float lists into Firestore Vector objects if present
        video_vec = (
            Vector(seg["video_embedding"])
            if seg.get("video_embedding")
            else None
        )
        text_vec = (
            Vector(seg["text_embedding"])
            if seg.get("text_embedding")
            else None
        )

        doc_data = {
            "video_id": seg["video_id"],
            "uid": seg["uid"],
            "segment_index": seg_idx,
            "start_sec": seg["start_sec"],
            "end_sec": seg["end_sec"],
            "transcript_text": seg["transcript_text"],
            "words": seg["words"],
            "video_embedding": video_vec,
            "text_embedding": text_vec,
            "created_at": get_utc_now(),
        }

        batch.set(doc_ref, doc_data)
        op_count += 1

        # Firestore batches allow up to 500 writes; commit every 400
        if op_count >= 400:
            batch.commit()
            batch = db.batch()
            op_count = 0

    if op_count > 0:
        batch.commit()


def load_transcript_checkpoint(
    uid: str,
    video_id: str
) -> tuple[str, list[dict[str, Any]]] | None:
    """Checks if Step 1 already saved transcript_checkpoint.json in GCS."""
    checkpoint_path = f"{uid}/lf_videos/{video_id}/transcript_checkpoint.json"
    bucket = gcs_client.bucket(GCS_ROOT_BUCKET_NAME)
    blob = bucket.blob(checkpoint_path)
    if not blob.exists():
        return None
    data = json.loads(blob.download_as_text())
    return data["transcript"], data["words"]


def save_transcript_checkpoint(
    uid: str,
    video_id: str,
    full_transcript: str,
    words: list[dict[str, Any]],
) -> None:
    """Saves Step 1 output to GCS so future runs skip Speech-to-Text."""
    checkpoint_path = f"{uid}/lf_videos/{video_id}/transcript_checkpoint.json"
    bucket = gcs_client.bucket(GCS_ROOT_BUCKET_NAME)
    blob = bucket.blob(checkpoint_path)
    blob.upload_from_string(
        json.dumps({"transcript": full_transcript, "words": words}),
        content_type="application/json",
    )


################################################################
# Core Pipeline: Transcribe -> Embed -> Index -> Update Status
################################################################

def process_video(video_id: str, uid: str) -> None:
    doc_ref = db.collection(LFVIDEO_COLLECTION_NAME).document(video_id)
    doc_snap = doc_ref.get()

    if not doc_snap.exists:
        raise RuntimeError(f"Video document '{video_id}' does not exist in Firestore.")

    video_data = doc_snap.to_dict()
    if video_data is None:
        raise RuntimeError(f"Video document '{video_id}' is empty.")

    if video_data.get("uid") != uid:
        raise PermissionError(f"UID mismatch for video '{video_id}'.")

    try:
        gcs_uri: str = video_data.get("gcs_uri", "")
        bucket_prefix = f"gs://{GCS_ROOT_BUCKET_NAME}/"
        if not gcs_uri.startswith(bucket_prefix):
            raise ValueError(f"Invalid gcs_uri: {gcs_uri}")

        video_blob_path = gcs_uri.removeprefix(bucket_prefix)
        mime_type: str = video_data.get("content_type") or "video/mp4"
        duration_sec: int = int(video_data.get("duration_seconds") or 30)

        # Load from GCS checkpoint if it exists, otherwise run Speech-to-Text V2
        checkpoint = load_transcript_checkpoint(uid=uid, video_id=video_id)
        if checkpoint is not None:
            full_transcript, words = checkpoint
            print(
                f"[1/4] Loaded saved transcript checkpoint from GCS "
                f"({len(words)} words) — skipping Speech-to-Text!",
                flush=True,
            )
        else:
            print(f"[1/4] Transcribing audio & word timestamps for video {video_id}...", flush=True)
            full_transcript, words = transcribe_video_from_gcs(
                uid=uid,
                video_id=video_id,
                video_blob_path=video_blob_path,
            )
            # Save checkpoint to GCS & Firestore immediately so we never re-run [1/4]
            save_transcript_checkpoint(uid, video_id, full_transcript, words)
            doc_ref.update({
                "transcript": full_transcript,
                "updated_at": get_utc_now(),
            })
            print("[1/4] Saved transcript checkpoint to GCS & Firestore.", flush=True)

        # If client didn't know duration, fallback to the last spoken word's timestamp
        if not video_data.get("duration_seconds") and words:
            duration_sec = max(duration_sec, int(words[-1]["end_sec"]) + 1)

        print(f"[2/4] Generating gemini-embedding-2 segments for video {video_id}...")
        segments = build_multimodal_segments(
            uid=uid,
            video_id=video_id,
            gcs_uri=gcs_uri,
            mime_type=mime_type,
            total_duration_sec=duration_sec,
            words=words,
        )

        print(f"[3/4] Saving {len(segments)} segments to Firestore...")
        save_segments_to_firestore(video_id=video_id, segments=segments)

        print(f"[4/4] Marking video {video_id} as SUCCESSFUL...")
        doc_ref.update({
            "status": "SUCCESSFUL",
            "is_complete": True,
            "transcript": full_transcript,
            "error_msg": None,
            "updated_at": get_utc_now(),
        })
        print(f"Done processing video {video_id}!")

    except Exception as e:
        print(f"ERROR processing video {video_id}: {e}")
        doc_ref.update({
            "status": "FAILED",
            "is_complete": False,
            "error_msg": f"Processing failed: {e}",
            "updated_at": get_utc_now(),
        })
        raise

################################################################
# CLI Entry point for Cloud Run Job
################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a long-form video for RAG.")
    parser.add_argument("--video-id", required=True, help="Firestore LFVideo document ID")
    parser.add_argument("--uid", required=True, help="Owner Firebase UID")
    args = parser.parse_args()

    process_video(video_id=args.video_id, uid=args.uid)