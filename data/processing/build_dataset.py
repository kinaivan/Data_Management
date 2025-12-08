# data/processing/build_dataset.py
from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from pymongo import MongoClient
from transformers import BertTokenizerFast

from .config import (
    MONGO_URI,
    LABEL_DB_NAME,
    LABEL_COLLECTION_NAME,
    JIRA_DB_NAME,
    HAS_LABEL_TAG,
    PROJECT_ECOSYSTEM_PREFIX,
    BERT_MODEL_NAME,
    MAX_SEQ_LENGTH,
    RAW_OUT,
    CLEAN_OUT,
    BERT_OUT,
    LABEL_MAP_OUT,
)
from .schemas import raw_schema, clean_schema

def normalize_tags(tags_field: Any) -> List[str]:
    if tags_field is None:
        return []
    if isinstance(tags_field, list):
        return [str(t) for t in tags_field]
    return [t.strip() for t in str(tags_field).split(",") if t.strip()]


def extract_ecosystem_from_tags(tags: List[str]) -> str | None:
    for t in tags:
        if t.startswith(PROJECT_ECOSYSTEM_PREFIX):
            return t.split("=", 1)[1]
    return None


def parse_global_id(global_id: str) -> Tuple[str, str]:
    parts = global_id.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected _id format in IssueLabels: {global_id}")
    return parts[0], parts[1]


def clean_jira_text(text: str | None) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Remove user mentions: [~username]
    text = re.sub(r"\[~[^\]]+\]", " ", text)

    # Remove link markup: [text|url]
    text = re.sub(r"\[([^|\]]+)\|[^\]]+\]", r"\1", text)

    # Remove code block markers {code} ... {code}
    text = re.sub(r"\{code[:]*[^\}]*\}", " ", text)

    # Remove panel/table markers
    text = re.sub(r"\{[a-zA-Z]+[:]*[^\}]*\}", " ", text)

    # Remove whitespace
    text = " ".join(text.split())
    return text


# ---------- Main pipeline steps ----------

def load_and_join_from_mongo(limit: int | None = None) -> pd.DataFrame:
    client = MongoClient(MONGO_URI)

    label_db = client[LABEL_DB_NAME]
    labels_coll = label_db[LABEL_COLLECTION_NAME]

    jira_db = client[JIRA_DB_NAME]

    # Get all labeled issues with "has-label" tag
    label_query = {"tags": HAS_LABEL_TAG}
    label_docs_cursor = labels_coll.find(label_query)
    if limit:
        label_docs_cursor = label_docs_cursor.limit(limit)
    label_docs = list(label_docs_cursor)

    records: List[Dict[str, Any]] = []

    for doc in label_docs:
        global_id = doc["_id"]
        tags_norm = normalize_tags(doc.get("tags"))

        # Determine collection + issue id
        coll_from_id, issue_id = parse_global_id(global_id)
        ecosystem = extract_ecosystem_from_tags(tags_norm) or coll_from_id

        jira_collection = jira_db[coll_from_id]
        issue = jira_collection.find_one({"id": issue_id})

        if issue is None:
            # No matching Jira issue
            continue

        fields = issue.get("fields", {})
        summary = fields.get("summary") or ""
        description = fields.get("description") or ""
        jira_key = issue.get("key")

        existence = bool(doc.get("existence", False))
        executive = bool(doc.get("executive", False))
        prop = bool(doc.get("property", False))
        has_add = existence or executive or prop

        # Ignore "predictions" as required
        records.append(
            {
                "global_id": str(global_id),
                "ecosystem": str(ecosystem),
                "issue_id": str(issue_id),
                "jira_key": str(jira_key) if jira_key else None,
                "summary": str(summary) if summary is not None else "",
                "description": str(description) if description is not None else "",
                "existence": existence,
                "executive": executive,
                "property": prop,
                "has_add": has_add,
                "tags": tags_norm,
            }
        )
    df_raw = pd.DataFrame.from_records(records)
    df_raw = raw_schema.validate(df_raw, lazy=True)
    return df_raw


def clean_and_build_text(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Clean Jira formatting from summary/description
    df = df_raw.copy()
    df["summary"] = df["summary"].apply(clean_jira_text)
    df["description"] = df["description"].apply(clean_jira_text)

    # Combine into single text field for BERT
    df["text"] = (
        df["summary"].fillna("") + " " + df["description"].fillna("")
    ).str.strip()

    # Drop empty texts
    df = df[df["text"].str.len() > 0]
    df_clean = df[
        ["global_id", "text", "has_add", "existence", "executive", "property"]
    ].copy()
    df_clean = clean_schema.validate(df_clean, lazy=True)
    return df_clean


def encode_binary_label(series: pd.Series) -> torch.Tensor:
    return torch.tensor(series.astype("int64").values, dtype=torch.long)


def bert_preprocess(df_clean: pd.DataFrame, out_path: str, label_map_path: str) -> None:
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    encodings = tokenizer(
        df_clean["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    y_tensor = encode_binary_label(df_clean["has_add"])
    dataset = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": y_tensor,
        "global_ids": df_clean["global_id"].tolist(),
    }
    torch.save(dataset, out_path)
    # label2id mapping
    label2id = {"no_add": 0, "has_add": 1}
    torch.save(label2id, label_map_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of labeled issues")
    parser.add_argument("--raw-out", type=str, default=RAW_OUT)
    parser.add_argument("--clean-out", type=str, default=CLEAN_OUT)
    parser.add_argument("--bert-out", type=str, default=BERT_OUT)
    parser.add_argument("--label-map-out", type=str, default=LABEL_MAP_OUT)
    args = parser.parse_args()

    # 1. Join labels + JiraRepos via id / tags
    df_raw = load_and_join_from_mongo(limit=args.limit)

    # 2. Clean text & build BERT input
    df_clean = clean_and_build_text(df_raw)

    # 3. Save to disk (for DVC)
    df_raw.to_parquet(args.raw_out, index=False)
    df_clean.to_parquet(args.clean_out, index=False)

    # 4. Build BERT-ready tensors
    bert_preprocess(df_clean, args.bert_out, args.label_map_out)


if __name__ == "__main__":
    main()