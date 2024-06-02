import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_embeddings(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    sentences = []
    embeddings = []
    for item in data:
        sentences.append(item['sentence'])
        embeddings.append(item['embedding'])
    return sentences, np.array(embeddings)

def plot_embeddings(sentences, embeddings):
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)
    for i, sentence in enumerate(sentences):
        plt.annotate(sentence, (embeddings[i, 0], embeddings[i, 1]))
    plt.title('Text Embeddings Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot text embeddings.')
    parser.add_argument('-m', '--model', choices=['openai', 'tf'], required=True, help='Choose the model: openai or tf')
    args = parser.parse_args()

    if args.model == 'openai':
        file_path = 'embeds_openai.json'
    elif args.model == 'tf':
        file_path = 'embeds_tf.json'
    
    sentences, embeddings = load_embeddings(file_path)
    plot_embeddings(sentences, embeddings)

if __name__ == "__main__":
    main()
