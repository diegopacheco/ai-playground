const tf = require('@tensorflow/tfjs-node');
const plot = require('node-remote-plot');

const sizes = tf.randomNormal([100, 1], 1000, 500);
const prices = sizes.mul(500).add(tf.randomNormal([100, 1], 0, 10000));

// Normalize sizes and prices
const {mean: sizesMean, variance: sizesVariance} = tf.moments(sizes);
const sizesStd = tf.sqrt(sizesVariance);
const {mean: pricesMean, variance: pricesVariance} = tf.moments(prices);
const pricesStd = tf.sqrt(pricesVariance);
const sizesNormalized = sizes.sub(sizesMean).div(sizesStd);
const pricesNormalized = prices.sub(pricesMean).div(pricesStd);

const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1], units: 1, kernelInitializer: 'heNormal'}));
model.compile({optimizer: tf.train.sgd(0.00001), loss: 'meanSquaredError'});

async function trainModel() {
    await model.fit(sizesNormalized, pricesNormalized, {epochs: 100});
    const predictions = model.predict(sizesNormalized);

    // Denormalize predictions
    const predictionsDenormalized = predictions.mul(pricesStd).add(pricesMean);

    plot({
        x: sizes.arraySync(),
        y: prices.arraySync(),
        x2: sizes.arraySync(),
        y2: predictionsDenormalized.arraySync(),
        title: 'Original data vs predictions',
        xLabel: 'Size',
        yLabel: 'Price'
    });
}

trainModel();