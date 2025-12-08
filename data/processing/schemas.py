import pandera as pa
from pandera import Column, Check
import pandas as pd


# Raw join between IssueLabels + Jira issue
raw_schema = pa.DataFrameSchema(
    {
        "global_id": Column(str, nullable=False),
        "ecosystem": Column(str, nullable=False),
        "issue_id": Column(str, nullable=False),
        "jira_key": Column(str, nullable=True),
        "summary": Column(str, nullable=True),
        "description": Column(str, nullable=True),
        "existence": Column(bool, nullable=False),
        "executive": Column(bool, nullable=False),
        "property": Column(bool, nullable=False),
        "has_add": Column(bool, nullable=False),
        "tags": Column(object, nullable=True),
    },
    strict=False,
)


# Cleaned text + final labels for model training
clean_schema = pa.DataFrameSchema(
    {
        "global_id": Column(str, nullable=False),
        "text": Column(str, nullable=False),
        "has_add": Column(bool, nullable=False),
        "existence": Column(bool, nullable=False),
        "executive": Column(bool, nullable=False),
        "property": Column(bool, nullable=False),
    },
    strict=True,
    checks=[
        Check(lambda df: df["text"].str.len() > 0, element_wise=False, error="Empty text"),
    ],
)