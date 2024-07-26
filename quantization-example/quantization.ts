export type Vector = number[];

const roundClip = (x: number, a: number, b: number) => {
  const value = Math.max(a, Math.min(b, Math.round(x)));
  return Object.is(value, -0) ? 0 : value;
};

export function quantizeWeights(W: Vector[]) {
  // Calculate average absolute value of weights
  let sumAbs = 0;
  const n = W.length; // Number of rows
  const m = W[0].length; // Number of columns

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      sumAbs += Math.abs(W[i][j]);
    }
  }

  const gamma = sumAbs !== 0 ? sumAbs / (n * m) : 0.000000000000001;

  // Quantize weights to {-1, 0, 1}
  const Wf: Vector[] = [];
  for (let i = 0; i < n; i++) {
    Wf[i] = [];

    for (let j = 0; j < m; j++) {
      Wf[i][j] = roundClip(W[i][j] / gamma, -1, 1);
    }
  }

  return Wf;
}
