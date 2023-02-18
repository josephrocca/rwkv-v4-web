function greedySampling(x, tokenHistory = undefined, repetition_penalty = 0) {
  x = applyRepetitionPenalty(x, tokenHistory, repetition_penalty)
  let max_k = 0;
  let max_v = x[0];
  const n = x.length;
  
  for (let i = 1; i < n; i++) {
    if (x[i] > max_v) {
      max_v = x[i];
      max_k = i;
    }
  }
  
  return max_k;
}

/** Given an array of probs that sum to 1, like:
 *   probs = [0.1, 0.25, 0.05, 0.6]
 * 
 *  Sort them in reverse order like:
 * 
 *   [0.6, 0.25, 0.1, 0.05]
 *
 *  Then find their cumulative probs like:
 *   [0.6, 0.85, 0.95, 1.0]
 *
 *  Then find the index which is greater than the top_p_usual.
 *   e.g if top_p_usual is 0.81 then that index is 1.
 *  Then return the prob in the original array for that index.
 *  e.g. 0.25
 *
 */
function find_cutoff(probs, top_p_usual) {
  let sorted_probs = probs.slice().sort((a,b) => b - a);
  for(let i = 0; i < sorted_probs.length - 1; ++i) {
    top_p_usual -= sorted_probs[i];
    if (top_p_usual <= 0) return sorted_probs[i];
  }
  return sorted_probs[sorted_probs.length-1];

  /** Above is equivalent to, but hopefully faster than: */
  /*
  let cumulative_probs = sorted_probs.reduce((a, x, i) => [...a, x + (a[i-1] || 0)], []);
  let cutoff = sorted_probs[cumulative_probs.findIndex(x => x > top_p_usual)];
  return cutoff
  */
}

function applyRepetitionPenalty(ozut, tokenHistory, repetition_penalty) {
  if (tokenHistory && repetition_penalty !== 0) {
    tokenHistory.forEach((token, i) =>
      ozut[token]=ozut[token] - repetition_penalty
    )
  }
  return ozut;
}

function getMultinomialProbs(ozut, temp = 1.0, top_p_usual = 0.8) {
  var probs = softmax(ozut);
  
  const cutoff = find_cutoff(probs, top_p_usual);
  probs = probs.map(x => x < cutoff ? 0 : x);
  if (temp !== 1) {
    probs = probs.map(x => Math.pow(x, 1.0 / temp));
  }
  
  const sum = probs.reduce((a, x) => a + x);
  return probs.map(x => x / sum);
}

// Given an array like:
//  ozut = [-1, 0, 3, 5]
// where this means that token 0 is least likely, token 1 next likely, then token 2, then token 3
// instead of just choosing token 3, we instead return one randomly, but making token the most likely etc.
function npsample(ozut, temp = 1.0, top_p_usual = 0.8, tokenHistory = undefined, repetition_penalty = 0) {  
  return choiceIndex(getMultinomialProbs(applyRepetitionPenalty(ozut, tokenHistory, repetition_penalty), temp, top_p_usual));
}

/**
 * This function generates a cumulative probability distribution from the input probabilities and then randomly selecting an index based on this distribution.
 * 
 * This is equivalent to:
 * 
 * 
 * let cumprobs = p.reduce((a, x, i) => [...a, x + (a[i-1] || 0)], []);
 * let x = Math.random();
 * let index = cumprobs.findIndex(y => y > x);
 * return index;
 *
 * But multiple orders of magnitude faster
 */
function choiceIndex(p) {
  const n = p.length;
  let x = Math.random();
  for (let i = 0; i < n; i++) {
    x -= p[i];
    if (x <= 0) return i;
  }
  return p[n-1]; // should never happen
}

function softmax(data, from = 0, to = data.length) {
  let max = -Infinity; // Math.max(...data) vould crash on large array
  for (let id = from; id < to; id++) {
    if (max < data[id]) {
      max = data[id];
    }
  }
  // No need to use reduce, just sum the exps in the loop
  let sumOfExp = 0;
  const result = Array.isArray(data) ? [] : {};
  for (let id = from; id < to; id++) {
    result[id] = Math.exp(data[id] - max);
    sumOfExp += result[id];
  }
  // Finally divide by the sum of exps
  for (let id = from; id < to; id++) {
    result[id] = result[id] / sumOfExp;
  }

  return result;
}

if (typeof module !== 'undefined') {
  module.exports = {greedySampling, npsample, softmax, choiceIndex, find_cutoff, getMultinomialProbs, applyRepetitionPenalty};
}
