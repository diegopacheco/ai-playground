const brain = require('brain.js');
const plot = require('nodeplotlib');

// Create a new neural network instance
const net = new brain.NeuralNetwork({ hiddenLayers: [3] });

// Train the neural network with weather data
// 1 represents sunny, 0 represents rainy
// Input is two days of temperatures (1 if above 70, 0 if below)
net.train([
  { input: [0, 0], output: [0] }, // two cold days -> rainy
  { input: [1, 0], output: [0] }, // one warm day, one cold day -> rainy
  { input: [0, 1], output: [0] }, // one cold day, one warm day -> rainy
  { input: [1, 1], output: [1] }  // two warm days -> sunny
]);

// Predict the weather for new data
const prediction00 = net.run([0, 0]); // Should be close to 0 (rainy)
const prediction01 = net.run([0, 1]); // Should be close to 0 (rainy)
const prediction10 = net.run([1, 0]); // Should be close to 0 (rainy)
const prediction11 = net.run([1, 1]); // Should be close to 1 (sunny)

// Plot the results
const trace = {
  x: ['[0, 0]', '[0, 1]', '[1, 0]', '[1, 1]'],
  y: [prediction00[0], prediction01[0], prediction10[0], prediction11[0]],
  type: 'bar'
};

const layout = {
  title: 'Weather Predictions',
  xaxis: { title: 'Two Days of Temperatures' },
  yaxis: { title: 'Predicted Weather (1 = Sunny, 0 = Rainy)' }
};

const data = [
  {
    x: ['[0, 0]', '[0, 1]', '[1, 0]', '[1, 1]'],
    y: [prediction00[0], prediction01[0], prediction10[0], prediction11[0]],
    type: 'bar'
  }
];
plot.plot(data);