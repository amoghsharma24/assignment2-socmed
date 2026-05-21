#!/usr/bin/env python3
"""
Assignment 2 YouTube data collection for the AI vs Human content study.

Example:
    python scripts/01_collection_A2.py --ai_file video_ids_ai.txt --human_file video_ids_human.txt --comments_per_video 1000

Outputs:
- data_raw/comments_master.csv
- data_raw/comments_master.json
- data_clean/network_edges.csv

The script extends the Assignment 1 collector by:
- preserving recursive YouTube pagination for top-level comments and replies
- adding video_type ("AI" or "Human") from the input video lists
- retaining parent_id for every reply
- building a reply network edge list for NetworkX analysis
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


MAX_RESULTS_PER_PAGE = 100
DEFAULT_COMMENTS_PER_VIDEO = 2000
VALID_VIDEO_TYPES = {"AI", "Human"}


@dataclass(frozen=True)
class VideoInput:
    video_id: str
    video_type: str
    source_value: str


def normalize_video_id(raw_value):
    """Extract a YouTube video ID from common YouTube ID and URL formats."""
    if not raw_value:
        return None

    value = raw_value.strip()
    if not value:
        return None

    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/shorts/([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    return None


def resolve_input_path(file_path, project_root):
    """Resolve input paths relative to the current directory, then the project root."""
    path = Path(file_path)
    if path.is_absolute() or path.exists():
        return path
    return project_root / path


def normalize_video_type(raw_value):
    """Normalise user-provided video labels to AI or Human."""
    if not raw_value:
        return None

    value = raw_value.strip().lower()
    if value in {"ai", "artificial", "artificial intelligence", "generated", "ai-generated"}:
        return "AI"
    if value in {"human", "human-created", "human created", "human-made", "creator"}:
        return "Human"
    return None


def deduplicate_video_inputs(video_inputs):
    """Remove duplicated video/type pairs while preserving order."""
    seen = set()
    deduped = []
    for item in video_inputs:
        key = (item.video_id, item.video_type)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def make_video_input(raw_value, video_type):
    """Normalize one CLI-provided video value into a typed video input."""
    video_id = normalize_video_id(raw_value)
    if not video_id:
        print(f"Skipping invalid {video_type} video input: {raw_value}")
        return None
    return VideoInput(video_id=video_id, video_type=video_type, source_value=raw_value)


def load_typed_video_file(file_path, fallback_video_type=None):
    """
    Load a video list file.

    Supported line formats:
    - https://www.youtube.com/watch?v=VIDEOID
    - VIDEOID
    - AI,https://www.youtube.com/watch?v=VIDEOID
    - Human,VIDEOID
    """
    path = Path(file_path)
    if not path.exists():
        return []

    video_inputs = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue

            raw_type = fallback_video_type
            raw_video = entry
            if "," in entry:
                first, second = [part.strip() for part in entry.split(",", 1)]
                parsed_type = normalize_video_type(first)
                if parsed_type:
                    raw_type = parsed_type
                    raw_video = second

            video_type = normalize_video_type(raw_type)
            video_id = normalize_video_id(raw_video)
            if not video_id or not video_type:
                print(f"Skipping invalid video input at {path}:{line_number}: {entry}")
                continue

            video_inputs.append(VideoInput(video_id=video_id, video_type=video_type, source_value=raw_video))

    return video_inputs


def load_manifest_csv(file_path):
    """Load a CSV manifest with video_id or video_url plus video_type columns."""
    path = Path(file_path)
    if not path.exists():
        return []

    df = pd.read_csv(path)
    columns = {column.lower().strip(): column for column in df.columns}
    video_column = columns.get("video_id") or columns.get("video_url") or columns.get("url")
    type_column = columns.get("video_type") or columns.get("type")

    if not video_column or not type_column:
        raise ValueError("Manifest CSV must include video_id/video_url and video_type columns.")

    video_inputs = []
    for row_number, row in df.iterrows():
        video_type = normalize_video_type(str(row[type_column]))
        raw_video = str(row[video_column])
        video_id = normalize_video_id(raw_video)
        if not video_id or not video_type:
            print(f"Skipping invalid manifest row {row_number + 2}: {raw_video}, {row[type_column]}")
            continue
        video_inputs.append(VideoInput(video_id=video_id, video_type=video_type, source_value=raw_video))

    return video_inputs


def get_author_channel_id(snippet):
    """Return the stable YouTube channel ID for a commenter when available."""
    author_channel = snippet.get("authorChannelId") or {}
    return author_channel.get("value")


def describe_http_error(error):
    """Return a compact status/reason string for a YouTube API HttpError."""
    status = getattr(error.resp, "status", "unknown")
    reason = getattr(error.resp, "reason", "")

    details = ""
    try:
        payload = json.loads(error.content.decode("utf-8"))
        errors = payload.get("error", {}).get("errors", [])
        if errors:
            details = errors[0].get("reason") or errors[0].get("message") or ""
        else:
            details = payload.get("error", {}).get("message", "")
    except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
        details = ""

    parts = [f"HTTP {status}"]
    if reason:
        parts.append(str(reason))
    if details and details != reason:
        parts.append(str(details))
    return " - ".join(parts)


def is_comments_disabled_error(error):
    """Detect the common YouTube API response for disabled comment threads."""
    try:
        payload = json.loads(error.content.decode("utf-8"))
    except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
        return False

    for item in payload.get("error", {}).get("errors", []):
        if item.get("reason") in {"commentsDisabled", "disabledComments"}:
            return True
    return False


def initialize_youtube(api_key):
    """Create the YouTube Data API v3 client."""
    return build("youtube", "v3", developerKey=api_key)


def get_video_metadata(youtube, video_id):
    """Fetch video title and channel fields for grouping and auditability."""
    try:
        request = youtube.videos().list(part="snippet,statistics", id=video_id)
        response = request.execute()
    except HttpError as error:
        print(f"  Could not fetch metadata for {video_id}: {describe_http_error(error)}")
        return None

    if not response.get("items"):
        print(f"  Video metadata not found for {video_id}. It may be private, deleted, or unavailable.")
        return None

    item = response["items"][0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    return {
        "video_title": snippet.get("title", "Unknown Title"),
        "channel_name": snippet.get("channelTitle", "Unknown Channel"),
        "video_published_at": snippet.get("publishedAt", ""),
        "video_view_count": statistics.get("viewCount"),
        "video_comment_count": statistics.get("commentCount"),
    }


def make_comment_record(
    *,
    item_id,
    snippet,
    video_input,
    metadata,
    parent_id,
    parent_author_id,
    is_reply,
    total_reply_count,
    collection_timestamp,
):
    """Flatten YouTube comment API payloads into analysis-ready rows."""
    author_display_name = snippet.get("authorDisplayName", "Unknown")
    author_channel_id = get_author_channel_id(snippet)
    return {
        "video_id": video_input.video_id,
        "video_type": video_input.video_type,
        "video_source_value": video_input.source_value,
        "video_title": metadata["video_title"],
        "channel_name": metadata["channel_name"],
        "video_published_at": metadata["video_published_at"],
        "video_view_count": metadata["video_view_count"],
        "video_comment_count": metadata["video_comment_count"],
        "comment_id": item_id,
        "parent_id": parent_id,
        "is_reply": is_reply,
        "author": author_display_name,
        "author_channel_id": author_channel_id,
        "parent_author_channel_id": parent_author_id,
        "comment_text": snippet.get("textDisplay", ""),
        "published_at": snippet.get("publishedAt", ""),
        "updated_at": snippet.get("updatedAt", ""),
        "like_count": snippet.get("likeCount", 0),
        "total_reply_count": total_reply_count,
        "collection_timestamp": collection_timestamp,
    }


def fetch_replies(youtube, parent_id, parent_author_id, video_input, metadata, collection_timestamp):
    """Fetch every available reply page for a top-level comment."""
    replies = []
    next_page_token = None
    page_number = 0

    while True:
        page_number += 1
        request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=MAX_RESULTS_PER_PAGE,
            pageToken=next_page_token,
            textFormat="plainText",
        )
        response = request.execute()
        page_items = response.get("items", [])
        print(f"    Reply page {page_number} for {parent_id}: {len(page_items)} replies")

        for item in page_items:
            snippet = item["snippet"]
            replies.append(
                make_comment_record(
                    item_id=item["id"],
                    snippet=snippet,
                    video_input=video_input,
                    metadata=metadata,
                    parent_id=parent_id,
                    parent_author_id=parent_author_id,
                    is_reply=True,
                    total_reply_count=0,
                    collection_timestamp=collection_timestamp,
                )
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return replies


def fetch_video_comments_and_replies(youtube, video_input, target_top_level_count, collection_timestamp):
    """Fetch top-level comments and all reply pages for one typed video."""
    comments_data = []
    metadata = get_video_metadata(youtube, video_input.video_id)
    if not metadata:
        return comments_data

    print("\n" + "=" * 70)
    print(f"Collecting {video_input.video_type} video: {video_input.video_id}")
    print(f"Title: {metadata['video_title']}")
    print(f"Channel: {metadata['channel_name']}")

    next_page_token = None
    top_level_count = 0
    page_number = 0

    while top_level_count < target_top_level_count:
        try:
            page_number += 1
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_input.video_id,
                maxResults=MAX_RESULTS_PER_PAGE,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            )
            response = request.execute()
            page_items = response.get("items", [])
            print(f"  Comment page {page_number}: {len(page_items)} top-level comments")

            for item in page_items:
                thread_snippet = item["snippet"]
                top_level_comment = thread_snippet["topLevelComment"]
                top_snippet = top_level_comment["snippet"]
                comment_id = top_level_comment["id"]
                reply_count = thread_snippet.get("totalReplyCount", 0)
                author_channel_id = get_author_channel_id(top_snippet)

                comments_data.append(
                    make_comment_record(
                        item_id=comment_id,
                        snippet=top_snippet,
                        video_input=video_input,
                        metadata=metadata,
                        parent_id=None,
                        parent_author_id=None,
                        is_reply=False,
                        total_reply_count=reply_count,
                        collection_timestamp=collection_timestamp,
                    )
                )
                top_level_count += 1

                if reply_count > 0:
                    try:
                        replies = fetch_replies(
                            youtube,
                            parent_id=comment_id,
                            parent_author_id=author_channel_id,
                            video_input=video_input,
                            metadata=metadata,
                            collection_timestamp=collection_timestamp,
                        )
                        comments_data.extend(replies)
                    except HttpError as error:
                        print(f"  Could not fetch replies for {comment_id}: {describe_http_error(error)}")

                if top_level_count >= target_top_level_count:
                    break

            print(f"  Fetched {top_level_count} top-level comments plus replies...")

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        except HttpError as error:
            status = getattr(error.resp, "status", None)
            if is_comments_disabled_error(error):
                print("  Comments are disabled for this video. Skipping gracefully.")
            elif status == 403:
                print(f"  Access forbidden, quota exceeded, or comments unavailable: {describe_http_error(error)}")
            elif status == 404:
                print(f"  Video or comments not found. It may be private, deleted, or restricted: {describe_http_error(error)}")
            elif status == 429:
                print(f"  Rate limit reached while collecting this video: {describe_http_error(error)}")
            else:
                print(f"  HTTP error while collecting {video_input.video_id}: {describe_http_error(error)}")
            break

    print(f"Completed {video_input.video_id}: {len(comments_data)} records")
    return comments_data


def build_network_edges(comments_df):
    """Build reply edges: reply author -> parent comment author."""
    reply_rows = comments_df[
        comments_df["is_reply"].eq(True)
        & comments_df["author_channel_id"].notna()
        & comments_df["parent_author_channel_id"].notna()
    ].copy()

    edges = reply_rows.rename(
        columns={
            "author_channel_id": "Source_User_ID",
            "parent_author_channel_id": "Target_User_ID",
            "comment_id": "Source_Comment_ID",
            "parent_id": "Target_Comment_ID",
        }
    )

    return edges[
        [
            "Source_User_ID",
            "Target_User_ID",
            "video_id",
            "video_type",
            "Source_Comment_ID",
            "Target_Comment_ID",
            "published_at",
        ]
    ]


def save_outputs(comments, project_root):
    """Persist the master comments dataset and the NetworkX edge list."""
    data_raw = project_root / "data_raw"
    data_clean = project_root / "data_clean"
    data_raw.mkdir(parents=True, exist_ok=True)
    data_clean.mkdir(parents=True, exist_ok=True)

    comments_df = pd.DataFrame(comments)
    comments_csv = data_raw / "comments_master.csv"
    comments_json = data_raw / "comments_master.json"
    edges_csv = data_clean / "network_edges.csv"

    comments_df.to_csv(comments_csv, index=False, encoding="utf-8")
    comments_df.to_json(comments_json, orient="records", indent=2, force_ascii=False)

    edges_df = build_network_edges(comments_df)
    edges_df.to_csv(edges_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 70)
    print("Assignment 2 data collection complete")
    print(f"Comments: {len(comments_df)} rows -> {comments_csv}")
    print(f"Edges: {len(edges_df)} rows -> {edges_csv}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect YouTube comments for AI vs Human comparison.",
        epilog=(
            "Example: python scripts/01_collection_A2.py --ai_videos VIDEO_ID "
            "--human_videos https://youtu.be/VIDEO_ID --comments_per_video 1000"
        ),
    )
    parser.add_argument("--ai_videos", nargs="+", help="AI-generated content YouTube IDs or URLs.")
    parser.add_argument("--human_videos", nargs="+", help="Human-created content YouTube IDs or URLs.")
    parser.add_argument("--ai_file", default="video_ids_ai.txt", help="Text file of AI video IDs/URLs.")
    parser.add_argument("--human_file", default="video_ids_human.txt", help="Text file of Human video IDs/URLs.")
    parser.add_argument("--manifest", help="CSV with video_id/video_url and video_type columns.")
    parser.add_argument(
        "--comments_per_video",
        type=int,
        default=DEFAULT_COMMENTS_PER_VIDEO,
        help=f"Top-level comments to collect per video. Replies are collected in addition. Default: {DEFAULT_COMMENTS_PER_VIDEO}.",
    )
    return parser.parse_args()


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    args = parse_args()

    load_dotenv(project_root / ".env")
    load_dotenv(Path.cwd() / ".env")
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found. Add it to Assignment2/.env or your current working directory .env.")

    video_inputs = []

    if args.manifest:
        video_inputs.extend(load_manifest_csv(resolve_input_path(args.manifest, project_root)))

    if args.ai_videos:
        video_inputs.extend(
            item for item in (make_video_input(value, "AI") for value in args.ai_videos) if item is not None
        )
    if args.human_videos:
        video_inputs.extend(
            item for item in (make_video_input(value, "Human") for value in args.human_videos) if item is not None
        )

    if not video_inputs:
        video_inputs.extend(load_typed_video_file(resolve_input_path(args.ai_file, project_root), fallback_video_type="AI"))
        video_inputs.extend(
            load_typed_video_file(resolve_input_path(args.human_file, project_root), fallback_video_type="Human")
        )

    video_inputs = deduplicate_video_inputs(video_inputs)
    if not video_inputs:
        raise ValueError(
            "No valid videos found. Use --ai_videos/--human_videos, --manifest, "
            "or create Assignment2/video_ids_ai.txt and Assignment2/video_ids_human.txt."
        )

    counts = pd.Series([item.video_type for item in video_inputs]).value_counts().to_dict()
    print("=" * 70)
    print("Starting Assignment 2 YouTube collection")
    print(f"Videos by type: {counts}")
    print(f"Top-level comment target per video: {args.comments_per_video}")
    print("=" * 70)

    youtube = initialize_youtube(api_key)
    collection_timestamp = datetime.now(timezone.utc).isoformat()

    all_comments = []
    for video_input in video_inputs:
        records = fetch_video_comments_and_replies(
            youtube=youtube,
            video_input=video_input,
            target_top_level_count=args.comments_per_video,
            collection_timestamp=collection_timestamp,
        )
        all_comments.extend(records)

    if not all_comments:
        print("No comments collected. Check video IDs, comment availability, and API quota.")
        return

    save_outputs(all_comments, project_root)


if __name__ == "__main__":
    main()
