This repository contains our implementation for the **Machine Learning Systems Deployment and Optimizations** assignment. The goal is to build a system that detects **Architectural Design Decisions (ADDs)** in Jira issues using machine learning.

### Current Status (Data Management & Preprocessing)

- **MongoDB integration**:
	- A MongoDB service is defined in `docker-compose.yml` (image `mongo:6`, exposed on `27017`).
	- The code expects the `JiraRepos` and `MiningDesignDecisions` databases to be restored from the provided dumps.

- **Label–issue join logic** (`build_dataset.py`):
	- Connects to MongoDB using `pymongo` with settings from `config.py`.
	- Reads from `MiningDesignDecisions.IssueLabels` and filters only entries with the `has-label` tag.
	- Parses the `_id` field to determine the Jira collection and issue id.
	- Looks up the corresponding issue in `JiraRepos.<collection>` using the `id` field.
	- Builds a raw dataframe containing `summary`, `description`, ADD label flags (`existence`, `executive`, `property`), a combined binary label `has_add`, and metadata such as `global_id`, `ecosystem`, and `jira_key`.
	- Ignores classifier `predictions` from the dataset, as required by the assignment.

- **Data validation with Pandera** (`data/processing/schemas.py`):
	- Validates the raw joined dataframe against a `raw_schema` to verify expected columns and types.
	- Validates the cleaned training dataframe against a stricter `clean_schema` that enforces non-empty text (`summary + description`) and consistent label columns.

- **Jira-specific text preprocessing**:
	- Implements `clean_jira_text` to remove Jira-specific markup such as user mentions (`[~user]`), link syntax (`[text|url]`), and `{code}` / `{panel}`-style macros.
	- Normalizes whitespace and converts non-string fields to strings.
	- Uses `summary` and `description` to build a single `text` field used as model input.

- **BERT-ready dataset creation**:
	- Uses `transformers.BertTokenizerFast` to tokenize the `text` field with a configurable model name and maximum sequence length.
	- Encodes the binary label `has_add` (`1` if any of `existence`, `executive`, or `property` is `True`).
	- Saves a PyTorch dictionary with `input_ids`, `attention_mask`, `labels`, and `global_ids` to `data/processed/add_bert.pt`, and a simple `label2id` mapping to `data/processed/label2id.pt`.

- **Output artefacts for data versioning**:
	- The preprocessing pipeline writes:
	- `data/raw/add_raw.parquet`: raw joined and validated data.
	- `data/clean/add_clean.parquet`: cleaned text + labels.
	- `data/processed/add_bert.pt`: BERT-ready tensors for training.
	- `data/processed/label2id.pt`: label mapping for the binary classifier.

### How to run the preprocessing pipeline (conda + DVC)

**1. Create / activate a conda environment**
  ```bash
  conda create -n mlsdo python=3.11    # or use an existing env
  conda activate mlsdo
  pip install -r requirements.txt
  ```

**2. Start MongoDB and restore the dumps (first time only)**
From the project root:
  ```bash
  docker-compose up -d mongo

  mongorestore --gzip --archive=mongodump-JiraRepos_2023-03-07-16\ 00.archive \
    --nsFrom="JiraRepos.*" --nsTo="JiraRepos.*"
  mongorestore --gzip --archive=mongodump-MiningDesignDecisions-lite.archive \
    --nsFrom="MiningDesignDecisions.*" --nsTo="MiningDesignDecisions.*"
  ```

**4. Run the pipeline via DVC**
With the same conda environment activated:
  ```bash
  dvc repro
  ```

This executes the `build_add_dataset` stage defined in `dvc.yaml`, generating:
- `data/raw/add_raw.parquet`
- `data/clean/add_clean.parquet`
- `data/processed/add_bert.pt`
- `data/processed/label2id.pt`# Data_Management
