import os

# Mongo connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# DB & collections
LABEL_DB_NAME = os.getenv("LABEL_DB_NAME", "MiningDesignDecisions")
LABEL_COLLECTION_NAME = os.getenv("LABEL_COLLECTION_NAME", "IssueLabels")

JIRA_DB_NAME = os.getenv("JIRA_DB_NAME", "JiraRepos")

# Tags
HAS_LABEL_TAG = os.getenv("HAS_LABEL_TAG", "has-label")
PROJECT_ECOSYSTEM_PREFIX = "project-ecosystem="

# BERT
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "bert-base-uncased")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256"))

# Output locations
RAW_OUT = os.getenv("RAW_OUT", "data/raw/add_raw.parquet")
CLEAN_OUT = os.getenv("CLEAN_OUT", "data/clean/add_clean.parquet")
BERT_OUT = os.getenv("BERT_OUT", "data/processed/add_bert.pt")
LABEL_MAP_OUT = os.getenv("LABEL_MAP_OUT", "data/processed/label2id.pt")