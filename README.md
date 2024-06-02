# Sentence Embedding and Clustering Tool

This project provides a simple tool for encoding sentences using OpenAI's embedding models and TensorFlow's Universal Sentence Encoder. It also implements k-means clustering to group sentences based on their embeddings.

## Features

- Encode sentences using OpenAI's embedding models and TensorFlow's Universal Sentence Encoder.
- Calculate Within-Cluster Sum of Squares (WCSS) to determine the optimal number of clusters for k-means clustering.
- Group sentences into clusters using k-means clustering algorithm.
- Save encoded sentences and clustered data into JSON files for future use.
- Plot embeddings using the provided Python script.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/AriManjikian/sentence-clustering.git
```

Navigate to the project directory:

```bash
cd sentence-clustering
```

Install dependencies:

```bash
npm install
```

```bash
pip install -r requirements.txt
```

## Usage

### Encoding Sentences

To encode sentences using OpenAI's embedding models, run:

```bash
node openai.js
```

To encode sentences using TensorFlow's Universal Sentence Encoder, run:

```bash
node tf.js
```

### Optional Flags

- Use the `-json` flag to skip the embedding process and retrieve embeddings from existing JSON files.

### Output

- The encoded sentences are saved in `embeds_openai.json` and `embeds_tf.json` respectively.
- Clusters are saved in `clusters_openai.json` and `clusters_tf.json`.

### Encoding Sentences

To plot embeddings using the provided Python script, run:

```bash
python plot.py -m {openai,tf}
```

## Examples

Encode sentences using OpenAI's models:

```bash
node openai.js
```

Encode sentences using TensorFlow's Universal Sentence Encoder:

```bash
node tf.js
```

Retrieve embeddings from JSON files without re-encoding:

```bash
node openai.js -json
```

Visualize embeddings from JSON files by plotting

```bash
python plot.py -m openai
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
