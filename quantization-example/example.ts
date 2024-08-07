import { quantizeWeights } from "./quantization";

// Example usage:
const weights = [
  [
    0.5670403838157654, 0.009260174818336964, 0.23178744316101074,
    -0.2916173040866852, -0.8924556970596313, 0.8785552978515625,
    -0.34576427936553955, 0.5742510557174683, -0.04222835972905159,
    -0.137906014919281,
  ],
];

const quantizedWeights = quantizeWeights(weights);
console.log("Quantized Weights:", quantizedWeights);
