#!/usr/bin/env python3
"""
Assignment 2 preprocessing adapter.

This keeps the Assignment 1 protected-term pattern, but replaces the F1 whitelist
with AI, art, music, and platform terms needed for the AI-generated vs Human-created
content comparison.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords


AI_TERMS = {
    "ai", "a.i", "artificial intelligence", "generative", "generative ai",
    "llm", "large language model", "midjourney", "sora", "dalle", "dall-e",
    "stable diffusion", "runway", "pika", "suno", "udio", "elevenlabs",
    "prompt", "prompting", "prompt engineering", "model", "diffusion",
    "text to image", "text to video", "image generation", "video generation",
    "synthetic media", "deepfake", "neural", "training data", "dataset",
}

ART_MUSIC_TERMS = {
    "artist", "creator", "illustrator", "animator", "composer", "producer",
    "musician", "vocal", "vocals", "voice", "lyrics", "melody", "harmony",
    "beat", "mix", "mastering", "instrumental", "animation", "painting",
    "drawing", "concept art", "cinematography", "editing", "render", "style",
    "aesthetic", "copyright", "licensing", "royalty", "original", "cover",
}

PLATFORM_TERMS = {
    "youtube", "shorts", "tiktok", "instagram", "reels", "spotify", "soundcloud",
    "views", "likes", "comments", "subscribe", "channel", "algorithm",
}

PROTECTED_TERMS = AI_TERMS | ART_MUSIC_TERMS | PLATFORM_TERMS
PROTECTED_TERM_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(re.escape(term) for term in sorted(PROTECTED_TERMS, key=len, reverse=True))
    + r")(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_HASHTAG_PATTERN = re.compile(r"@\w+|#\w+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+"
)
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def protected_token(term):
    """Convert a protected phrase into a single analysis token."""
    return re.sub(r"\W+", "_", term.lower()).strip("_")


PROTECTED_TOKENS = {protected_token(term) for term in PROTECTED_TERMS}
STOP_WORDS = None


def ensure_nltk_resources():
    for resource, download_name in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(download_name)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = MENTION_HASHTAG_PATTERN.sub("", text)
    text = EMOJI_PATTERN.sub("", text)
    text = PROTECTED_TERM_PATTERN.sub(lambda match: protected_token(match.group(0)), text)
    text = PUNCTUATION_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOP_WORDS or word in PROTECTED_TOKENS]


def build_stop_words():
    """Build the reusable stopword set once instead of once per row."""
    custom_stopwords = {
        "lol", "omg", "lmao", "haha", "yeah", "ok", "guys", "guy", "man",
        "dude", "bro", "like", "actually", "just", "get", "got", "getting",
        "think", "thing", "things", "probably", "maybe", "says", "say",
        "said", "saying", "people", "person", "will", "make", "see", "going",
        "really", "even", "video",
    }
    return set(stopwords.words("english")) | custom_stopwords


def plot_top_words(tokens, output_path, top_n=50):
    counts = Counter(tokens)
    top_words = counts.most_common(top_n)
    if not top_words:
        print("No words available for frequency plot.")
        return

    words, values = zip(*top_words)
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(words)), values, color="steelblue", edgecolor="black")
    plt.yticks(range(len(words)), words, fontsize=10)
    plt.xlabel("Frequency")
    plt.title("Top 50 Most Frequent Words")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Assignment 2 YouTube comments.")
    parser.add_argument("--input", default="../data_raw/comments_master.csv", help="Input comments CSV.")
    parser.add_argument("--output", default="../data_clean/comments_master_processed.csv", help="Output processed CSV.")
    return parser.parse_args()


def main():
    global STOP_WORDS

    ensure_nltk_resources()
    STOP_WORDS = build_stop_words()
    script_dir = Path(__file__).resolve().parent
    args = parse_args()

    input_path = (script_dir / args.input).resolve()
    output_path = (script_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} comments from {input_path}")
    df["cleaned_text"] = df["comment_text"].apply(clean_text)
    df["tokens"] = df["cleaned_text"].apply(tokenize)
    df["tokens_filtered"] = df["tokens"].apply(remove_stopwords)
    df["tokens_str"] = df["tokens_filtered"].apply(lambda tokens: " ".join(tokens))
    df.to_csv(output_path, index=False, encoding="utf-8")

    all_tokens = [token for tokens in df["tokens_filtered"] for token in tokens]
    plot_top_words(all_tokens, output_path.parent / "top_50_words_A2.png")

    print(f"Processed {len(df)} comments -> {output_path}")
    print(f"Protected terms loaded: {len(PROTECTED_TERMS)}")


if __name__ == "__main__":
    main()
