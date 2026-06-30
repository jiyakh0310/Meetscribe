# ML Pipeline

## Purpose

Document the current Machine Learning based Minutes of Meeting pipeline modules as they are implemented incrementally. The clustering stage groups semantically similar transcript sentences using sentence embeddings and prepares topic-like sentence groups for later MoM generation.

## Inputs

The clustering module receives `SentenceEmbedding` objects from `ml_mom/embeddings.py`.

Each embedding contains:

- `sentence_id`
- `turn_id`
- `speaker`
- `original_sentence`
- `embedding_vector`
- `embedding_dimension`

## Outputs

The clustering module returns `SentenceCluster` objects through a `ClusteringResult`.

Each cluster contains:

- `cluster_id`
- `topic_label`
- `sentence_ids`
- `member_sentences`
- `centroid_embedding`
- `cluster_size`

## Current Status

- `ml_mom/transcript_parser.py`: completed.
- `ml_mom/preprocessing.py`: completed.
- `ml_mom/feature_extraction.py`: completed.
- `ml_mom/embeddings.py`: completed with graceful unavailable-model handling.
- `ml_mom/clustering.py`: completed for clustering only, with Agglomerative as the default strategy and KMeans, DBSCAN, and HDBSCAN strategy methods available.
- `ml_mom/annotation_tool.py`: completed as a non-production dataset annotation utility.
- `ml_mom/training_dataset.py`: completed as a dataset preparation utility for future ANN training.
- `ml_mom/ann_model.py`: completed as the PyTorch ANN architecture definition only.

## Annotation Stage

### Purpose

The annotation stage creates supervised training data for the future ANN sentence classifier. It lets a human label parsed transcript sentences as `Discussion`, `Decision`, `Action_Item`, `Summary`, or `Information`.

### Inputs

The annotation tool accepts either:

- A transcript text file parsed through `ml_mom/transcript_parser.py`.
- Existing parsed `TranscriptTurn` objects.

### Outputs

The annotation tool exports sentence-level training rows to CSV and JSON.

Each row contains:

- `sentence_id`
- `turn_id`
- `speaker`
- `timestamp`
- `sentence`
- `selected_label`
- `cluster_id`
- `notes`

### Future Integration

- Keep the annotation tool outside the production Streamlit workflow.
- Add a GUI annotation screen only after the dataset schema stabilizes.
- Use annotated CSV or JSON files later for ANN training and validation.
- Optionally attach cluster IDs from `ml_mom/clustering.py` once clustering quality is validated.

## Training Dataset Stage

### Purpose

The training dataset stage prepares annotated CSV files for future ANN training. It validates annotation exports, removes unusable rows, encodes labels into numeric class IDs, and creates train/validation/test splits.

### Inputs

The stage accepts CSV files exported by `ml_mom/annotation_tool.py`.

Required columns:

- `sentence_id`
- `turn_id`
- `speaker`
- `timestamp`
- `sentence`
- `selected_label`
- `cluster_id`
- `notes`

### Outputs

The stage returns a `PreparedTrainingDataset` containing:

- cleaned samples
- numeric label IDs
- training samples
- validation samples
- testing samples
- label mapping
- dataset statistics
- validation messages

### Future Integration

- Use this prepared dataset as input to the future ANN classifier training script.
- Keep training separate from dataset preparation.
- Add class balancing and data augmentation only after real annotation statistics are reviewed.
- Preserve the production Streamlit, export, email, and SMTP flows while dataset tooling evolves.

## ANN Model Stage

### Purpose

The ANN Model stage defines the neural network architecture that will later classify transcript sentence embeddings into MoM categories.

### Inputs

The model accepts sentence embedding vectors. The default embedding dimension is `384`, matching `all-MiniLM-L6-v2`.

### Outputs

The model produces logits, probability distributions, and predicted class labels for:

- `Discussion`
- `Decision`
- `Action_Item`
- `Summary`
- `Information`

### Current Status

`ml_mom/ann_model.py` defines a PyTorch feed-forward ANN with:

- input layer
- hidden layer 1 with ReLU
- dropout
- hidden layer 2 with ReLU
- dropout
- output layer

It performs no training, dataset loading, checkpoint saving, or production integration.

### Future Integration

- Add a separate training script that consumes `PreparedTrainingDataset`.
- Add BiLSTM support if sentence sequence context becomes necessary.
- Add Transformer encoder support if richer contextual modeling is needed.
- Add hyperparameter tuning after baseline model metrics are available.

## Future Integration Notes

- Do not connect clustering to the Streamlit UI until the complete ML pipeline is validated.
- Do not use clustering output for ANN classification until the classifier module is implemented and tested.
- Replace temporary keyword-based topic labels with stronger noun-phrase extraction after real meeting transcript samples are available.
- Add quality metrics such as silhouette score after real embeddings are produced.
- Preserve the existing PDF, DOCX, SMTP, speaker review, and transcript review flows during future integration.
