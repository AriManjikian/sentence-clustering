const tf = require('@tensorflow/tfjs');
const use = require('@tensorflow-models/universal-sentence-encoder');
const fs = require('fs');
const { kmeans } = require('ml-kmeans');
const { agnes } = require('ml-hclust');

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

// Hierarchical Agglomerative Clustering (HAC)
const performHAC = (data, k) => {
    const clusters = agnes(data);
    return clusters.cut(k);
};

// Simple DBSCAN implementation
const performDBSCAN = (data, epsilon, minPts) => {
    const clusters = [];
    const visited = new Set();

    function regionQuery(point, epsilon) {
        return data.reduce((neighbors, other, index) => {
            if (euclideanDistance(point, other) < epsilon) {
                neighbors.push(index);
            }
            return neighbors;
        }, []);
    }

    function expandCluster(point, neighbors, cluster) {
        cluster.push(point);
        visited.add(point);

        for (let i = 0; i < neighbors.length; i++) {
            const neighbor = neighbors[i];
            if (!visited.has(neighbor)) {
                visited.add(neighbor);
                const newNeighbors = regionQuery(data[neighbor], epsilon);
                if (newNeighbors.length >= minPts) {
                    neighbors.push(...newNeighbors.filter(n => !neighbors.includes(n)));
                }
            }
            if (!clusters.some(c => c.includes(neighbor))) {
                cluster.push(neighbor);
            }
        }
    }

    for (let i = 0; i < data.length; i++) {
        if (visited.has(i)) continue;
        const neighbors = regionQuery(data[i], epsilon);
        if (neighbors.length < minPts) {
            visited.add(i);
        } else {
            const cluster = [];
            expandCluster(i, neighbors, cluster);
            clusters.push(cluster);
        }
    }

    return data.map((_, index) => {
        const clusterIndex = clusters.findIndex(cluster => cluster.includes(index));
        return clusterIndex === -1 ? -1 : clusterIndex;
    });
};

function euclideanDistance(a, b) {
    return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
}

// Simple Gaussian Mixture Model implementation
const performGMM = (data, k, iterations = 100) => {
    // Initialize means randomly
    let means = data.slice(0, k);
    let covariances = Array(k).fill().map(() => Array(data[0].length).fill().map(() => Math.random()));
    let weights = Array(k).fill(1 / k);

    for (let iter = 0; iter < iterations; iter++) {
        // E-step: Calculate responsibilities
        const responsibilities = data.map(point => {
            const densities = means.map((mean, j) => {
                const diff = point.map((val, i) => val - mean[i]);
                const exponent = diff.reduce((sum, d, i) => sum + d * d / covariances[j][i], 0);
                return weights[j] * Math.exp(-0.5 * exponent) / Math.sqrt(2 * Math.PI * covariances[j].reduce((a, b) => a * b, 1));
            });
            const total = densities.reduce((a, b) => a + b, 0);
            return densities.map(d => d / total);
        });

        // M-step: Update parameters
        for (let j = 0; j < k; j++) {
            const nk = responsibilities.reduce((sum, r) => sum + r[j], 0);
            weights[j] = nk / data.length;
            means[j] = data.reduce((sum, point, i) => sum.map((s, dim) => s + responsibilities[i][j] * point[dim]), Array(data[0].length).fill(0))
                .map(s => s / nk);
            covariances[j] = data[0].map((_, dim) =>
                data.reduce((sum, point, i) => sum + responsibilities[i][j] * Math.pow(point[dim] - means[j][dim], 2), 0) / nk
            );
        }
    }

    // Assign each point to the cluster with highest responsibility
    return data.map(point => {
        const densities = means.map((mean, j) => {
            const diff = point.map((val, i) => val - mean[i]);
            const exponent = diff.reduce((sum, d, i) => sum + d * d / covariances[j][i], 0);
            return weights[j] * Math.exp(-0.5 * exponent) / Math.sqrt(2 * Math.PI * covariances[j].reduce((a, b) => a * b, 1));
        });
        return densities.indexOf(Math.max(...densities));
    });
};

(async () => {
    const embedType = process.argv[2];
    const sentences = require('./sentences');
    const clusteringAlgorithm = process.argv[3] || 'kmeans';
    let clusters;
    let optimalClusters;
    // Get array of embeddings
    let embeddingsArr = [];
    if (embedType == "-json") {
        embeddingsArr = require("./embeds_tf.json").map(item => item.embedding);
    } else {
        await tf.setBackend('cpu');
        const model = await use.load();
        const embeddings = await model.embed(sentences);
        embeddingsArr = await embeddings.arraySync();
        // Combine sentences and their embeddings
        const combinedData = sentences.map((sentence, index) => ({
            sentence: sentence,
            embedding: embeddingsArr[index]
        }));

        // Write combined sentences and embeddings to a file
        fs.writeFileSync('embeds_tf.json', JSON.stringify(combinedData, null, 2));
        console.log('Embeddings have been written to embeds_tf.json');
    }

    switch (clusteringAlgorithm) {
        case 'kmeans':
            const maxClusters = Math.min(10, embeddingsArr.length); // Maximum number of clusters to evaluate
            const wcssValues = [];
            for (let k = 1; k <= maxClusters; k++) {
                const wcss = calculateWCSS(embeddingsArr, k);
                wcssValues.push(wcss);
            }

            optimalClusters = calculateElbowPoint(wcssValues);
            console.log(`Optimal number of clusters determined by elbow method: ${optimalClusters}`);

            // Perform K-means clustering with the optimal number of clusters
            clusters = kmeans(embeddingsArr, optimalClusters, {
                initialization: "random",

            }).clusters;
            break;

        case 'hac':
            optimalClusters = 5; // You may want to implement a method to determine this
            clusters = performHAC(embeddingsArr, optimalClusters);
            console.log(`Performed HAC clustering with ${optimalClusters} clusters`);
            break;

        case 'dbscan':
            const epsilon = 0.5; // You may need to adjust these parameters
            const minPts = 3;
            clusters = performDBSCAN(embeddingsArr, epsilon, minPts);
            optimalClusters = Math.max(...clusters) + 1;
            console.log(`Performed DBSCAN clustering. Found ${optimalClusters} clusters`);
            break;

        case 'gmm':
            optimalClusters = 5; // You may want to implement a method to determine this
            clusters = performGMM(embeddingsArr, optimalClusters);
            console.log(`Performed GMM clustering with ${optimalClusters} clusters`);
            break;

        default:
            console.error('Unknown clustering algorithm');
            process.exit(1);
    }

    // Initialize groups
    const groups = Array.from({ length: optimalClusters }, (_, i) => ({
        group: `group ${i + 1}`,
        sentences: []
    }));

    // Assign each sentence to a group based on the cluster result
    sentences.forEach((sentence, index) => {
        const cluster = clusters[index];
        groups[cluster].sentences.push(sentence);
    });

    // Write grouped sentences to a file
    fs.writeFileSync('clusters_tf.json', JSON.stringify(groups, null, 2));
    console.log('Clusters have been written to clusters_tf.json');
})();
