#!/usr/bin/env python3
"""
Create balanced AI vs Human analysis files without modifying the full dataset.

The Human sample is selected at the thread level, so top-level comments and their
replies stay together. This keeps the reply network auditable while making the
Human row count close to the AI row count.
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SEED = 42


def thread_key(frame):
    """Return the top-level comment ID for each row."""
    return frame["parent_id"].fillna(frame["comment_id"])


def sample_human_threads(human_df, target_rows, seed):
    """Sample Human threads until the total row count is close to target_rows."""
    human_df = human_df.copy()
    human_df["_thread_id"] = thread_key(human_df)

    thread_sizes = human_df.groupby("_thread_id").size().sample(frac=1, random_state=seed)
    selected_threads = []
    selected_rows = 0

    for thread_id, size in thread_sizes.items():
        if selected_rows >= target_rows:
            break

        overage_without = abs(target_rows - selected_rows)
        overage_with = abs(target_rows - (selected_rows + size))

        if selected_rows == 0 or selected_rows + size <= target_rows or overage_with < overage_without:
            selected_threads.append(thread_id)
            selected_rows += size

    return human_df[human_df["_thread_id"].isin(selected_threads)].drop(columns=["_thread_id"])


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


def parse_args():
    parser = argparse.ArgumentParser(description="Create balanced Assignment 2 analysis datasets.")
    parser.add_argument("--raw-input", default="../data_raw/comments_master.csv")
    parser.add_argument("--processed-input", default="../data_clean/comments_master_processed.csv")
    parser.add_argument("--raw-output", default="../data_clean/comments_master_balanced.csv")
    parser.add_argument("--processed-output", default="../data_clean/comments_master_processed_balanced.csv")
    parser.add_argument("--edges-output", default="../data_clean/network_edges_balanced.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    script_dir = Path(__file__).resolve().parent
    args = parse_args()

    raw_input = (script_dir / args.raw_input).resolve()
    processed_input = (script_dir / args.processed_input).resolve()
    raw_output = (script_dir / args.raw_output).resolve()
    processed_output = (script_dir / args.processed_output).resolve()
    edges_output = (script_dir / args.edges_output).resolve()

    raw_df = pd.read_csv(raw_input)
    processed_df = pd.read_csv(processed_input)

    ai_df = raw_df[raw_df["video_type"].eq("AI")]
    human_df = raw_df[raw_df["video_type"].eq("Human")]
    sampled_human_df = sample_human_threads(human_df, target_rows=len(ai_df), seed=args.seed)

    balanced_raw = (
        pd.concat([ai_df, sampled_human_df], ignore_index=True)
        .sort_values(["video_type", "video_id", "published_at", "comment_id"], kind="stable")
        .reset_index(drop=True)
    )

    balanced_ids = set(balanced_raw["comment_id"])
    balanced_processed = (
        processed_df[processed_df["comment_id"].isin(balanced_ids)]
        .sort_values(["video_type", "video_id", "published_at", "comment_id"], kind="stable")
        .reset_index(drop=True)
    )
    balanced_edges = build_network_edges(balanced_raw)

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    balanced_raw.to_csv(raw_output, index=False, encoding="utf-8")
    balanced_processed.to_csv(processed_output, index=False, encoding="utf-8")
    balanced_edges.to_csv(edges_output, index=False, encoding="utf-8")

    print("Balanced dataset complete")
    print(f"AI rows kept: {len(ai_df)}")
    print(f"Human rows sampled: {len(sampled_human_df)}")
    print(f"Balanced raw rows: {len(balanced_raw)} -> {raw_output}")
    print(f"Balanced processed rows: {len(balanced_processed)} -> {processed_output}")
    print(f"Balanced network edges: {len(balanced_edges)} -> {edges_output}")
    print(f"Random seed: {args.seed}")


if __name__ == "__main__":
    main()
