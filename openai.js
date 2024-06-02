require('dotenv').config();
const { default: OpenAI } = require("openai");
const fs = require('fs');
const { kmeans } = require("ml-kmeans");

const openai = new OpenAI({
    apiKey: process.env.OPENAI_KEY
});

// Embed given text
async function embedText(text) {
    try {
        const response = await openai.embeddings.create({
            model: "text-embedding-3-small", // Change embedding model
            input: text
        });
        return response;
    } catch (error) {
        console.error(`Error generating embedding for text: "${text}"`, error);
    }
}

// Process embeddings for each sentence
async function processSentences(sentences) {
    const embeddings = [];

    for (const sentence of sentences) {
        let res = await embedText(sentence);
        if (res) {
            const embedding = res.data[0].embedding;
            embeddings.push({ sentence, embedding });
        }
    }

    fs.writeFileSync('embeds_openai.json', JSON.stringify(embeddings, null, 2));
    console.log('Embeddings have been written to embeds_openai.json');
    return embeddings;
}

// Calculate the WCSS for different numbers of clusters
const calculateWCSS = (data, k) => {
    const kmeansResult = kmeans(data, k);
    let wcss = 0;
    for (let i = 0; i < k; i++) {
        const clusterPoints = data.filter((_, idx) => kmeansResult.clusters[idx] === i);
        const centroid = kmeansResult.centroids[i];
        const distances = clusterPoints.map(point =>
            Math.sqrt(point.reduce((sum, val, j) => sum + Math.pow(val - centroid[j], 2), 0))
        );
        wcss += distances.reduce((sum, val) => sum + val, 0);
    }
    return wcss;
};

// Determine the elbow point
const calculateElbowPoint = (wcssValues) => {
    const diffs = [];
    for (let i = 1; i < wcssValues.length; i++) {
        diffs.push(wcssValues[i - 1] - wcssValues[i]);
    }
    let maxDiff = 0;
    let elbowPoint = 1;
    for (let i = 1; i < diffs.length; i++) {
        const diffChange = diffs[i - 1] - diffs[i];
        if (diffChange > maxDiff) {
            maxDiff = diffChange;
            elbowPoint = i + 1;
        }
    }
    return elbowPoint;
};

(async () => {
    const embedType = process.argv[2]
    const sentences = require('./sentences');

    // Get array of embeddings
    let embeddingsArr = []
    if (embedType == "-json") {
        embeddingsArr = require("./embeds_openai.json");
    } else {
        embeddingsArr = await processSentences(sentences);
    }
    embeddingsArr = embeddingsArr.map(item => item.embedding);

    const maxClusters = 10;
    const wcssValues = [];
    for (let k = 1; k <= maxClusters; k++) {
        const wcss = calculateWCSS(embeddingsArr, k);
        wcssValues.push(wcss);
    }

    const optimalClusters = calculateElbowPoint(wcssValues);
    console.log(`Optimal number of clusters determined by elbow method: ${optimalClusters}`);

    // Perform K-means clustering with the optimal number of clusters
    const kmeansResult = kmeans(embeddingsArr, optimalClusters);

    // Initialize groups
    const groups = Array.from({ length: optimalClusters }, (_, i) => ({
        group: `group ${i + 1}`,
        sentences: []
    }));

    // Assign each sentence to a group based on the cluster result
    sentences.forEach((sentence, index) => {
        const cluster = kmeansResult.clusters[index];
        groups[cluster].sentences.push(sentence);
    });

    // Write grouped sentences to a file
    fs.writeFileSync('clusters_openai.json', JSON.stringify(groups, null, 2));
    console.log('Clusters have been written to clusters_openai.json');
})();

