const tf = require('@tensorflow/tfjs-node-gpu');
const {layerNormalization} = require('./layers');

function selfAttention(inputs, dModel) {
  // Inputs: [batchSize, sequenceLength, dModel]
  const Q = tf.layers.dense({
    units: dModel,
    activation: 'linear',
    name: 'query',
  }).apply(inputs); // [batchSize, sequenceLength, dModel]

  const K = tf.layers.dense({
    units: dModel,
    activation: 'linear',
    name: 'key',
  }).apply(inputs); // [batchSize, sequenceLength, dModel]

  const V = tf.layers.dense({
    units: dModel,
    activation: 'linear',
    name: 'value',
  }).apply(inputs); // [batchSize, sequenceLength, dModel]

  // Compute dot product attention scores
  const scores = tf.matMul(Q, K.transpose([0, 2, 1])); // [batchSize, sequenceLength, sequenceLength]

  // Scale attention scores by dimensionality of key vectors
  const dk = tf.scalar(dModel).sqrt();
  const scaledScores = tf.div(scores, dk);

  // Compute attention weights using softmax
  const attentionWeights = tf.softmax(scaledScores, -1); // [batchSize, sequenceLength, sequenceLength]

  // Apply attention weights to value vectors
  const output = tf.matMul(attentionWeights, V); // [batchSize, sequenceLength, dModel]

  // Add skip connection and layer normalization
  const outputWithSkip = tf.add(output, inputs);
  const outputNormalized = layerNormalization(outputWithSkip);

  return outputNormalized;
}

// Test the self-attention function
const batchSize = 1;
const sequenceLength = 4;
const dModel = 8;
const inputs = tf.randomNormal([batchSize, sequenceLength, dModel]);
const output = selfAttention(inputs, dModel);
console.log(output.shape);
