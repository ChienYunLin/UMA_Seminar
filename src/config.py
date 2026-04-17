"""
Shared path configuration for the link-prediction analysis notebooks.
"""
from pathlib import Path

BASE_DIR    = Path("..")  # project root
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR  = BASE_DIR / "analysis_output"   # project-root output (not scripts/)

# ── Result directories ──────────────────────────────────────────────────────
# NOTE: "user_mention_reply_orig" is a PLACEHOLDER — available after upload.
#       All analyses guard with available_result_dirs() before accessing it.
RESULT_DIRS = {
    "tweet_mention_orig":       RESULTS_DIR / "results_not_permutated" / "mention_link_prediction",
    "tweet_mention_perm":       RESULTS_DIR / "results_permutated"     / "mention_link_prediction",
    "user_mention_orig":        RESULTS_DIR / "results_not_permutated" / "user_mention_link_prediction",
    "user_mention_perm":        RESULTS_DIR / "results_permutated"     / "user_mention_link_prediction",
    "user_mention_reply_orig":  RESULTS_DIR / "results_not_permutated" / "user_mention_link_prediction_with_reply",  # PLACEHOLDER
    "user_mention_reply_perm":  RESULTS_DIR / "results_permutated"     / "user_mention_link_prediction_with_reply",
}

# ── Data files ───────────────────────────────────────────────────────────────
MENTION_REL_PATH       = DATA_DIR / "mention_rel.parquet"
REPLY_MENTION_REL_PATH = DATA_DIR / "reply_mention_rel.parquet"
USERS_PATH             = DATA_DIR / "users.parquet"
TWEETS_PATH            = DATA_DIR / "tweets.parquet"
COMMUNITY_CSV          = DATA_DIR / "community_detection" / "user_communities_leiden.csv"
COMMUNITY_STATS_CSV    = DATA_DIR / "community_detection" / "community_statistics_leiden.csv"
COMMUNITY_TYPES_CSV    = DATA_DIR / "community_detection" / "community_types_id.csv"

# ── Temporal split timestamps ────────────────────────────────────────────────
VAL_TIMESTAMP  = "2019-09-01"
TEST_TIMESTAMP = "2019-11-01"

# ── Model config ─────────────────────────────────────────────────────────────
N_RUNS = 5
SEEDS  = [42, 123, 456, 789, 1024]
EVAL_K = 10

# Human-readable label for each result directory key
CONDITION_LABELS = {
    "tweet_mention_orig":       "TweetMention / Original",
    "tweet_mention_perm":       "TweetMention / Permuted",
    "user_mention_orig":        "UserMention / Original",
    "user_mention_perm":        "UserMention / Permuted",
    "user_mention_reply_orig":  "UserMention+Reply / Original",   # PLACEHOLDER
    "user_mention_reply_perm":  "UserMention+Reply / Permuted",
}


def available_result_dirs() -> dict:
    """
    Return the subset of RESULT_DIRS whose results.json file actually exists.
    Use instead of RESULT_DIRS to gracefully skip placeholders not yet uploaded.
    """
    return {k: v for k, v in RESULT_DIRS.items() if (v / "results.json").exists()}
