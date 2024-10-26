use rand::Rng;
use std::ops::Range;

const EPS: f32 = 1e-5;

// --------------------------------------------------------------------------------
// vector +-*/ scalar

pub fn add(input: &[f32], a: f32) -> Vec<f32> {
    input.iter().map(|input| input + a).collect()
}
pub fn sub(input: &[f32], a: f32) -> Vec<f32> {
    input.iter().map(|input| input - a).collect()
}
pub fn mul(input: &[f32], a: f32) -> Vec<f32> {
    input.iter().map(|input| input * a).collect()
}
pub fn div(input: &[f32], a: f32) -> Vec<f32> {
    input.iter().map(|input| input / a).collect()
}

// --------------------------------------------------------------------------------
// vector +-*/ vector

pub fn add_v(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| *a + *b).collect()
}
pub fn sub_v(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| *a - *b).collect()
}
pub fn mul_v(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| *a * *b).collect()
}
pub fn div_v(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| *a / *b).collect()
}

// --------------------------------------------------------------------------------
// Reduce Operators: Vector -> Scalar

pub fn max(input: &[f32]) -> (usize, f32) {
    let (max_idx, max_val) = input
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (max_idx, *max_val)
}

pub fn sum(input: &[f32]) -> f32 {
    let mut total = 0.0;
    for &val in input {
        total += val;
    }
    total
}

// --------------------------------------------------------------------------------
// Matrix

// Inner Product
pub fn inner(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
}

// Matrix Multiply
pub fn matmul(input: &[f32], w: &[Vec<f32>]) -> Vec<f32> {
    w.iter()
        .enumerate()
        .map(|(_, row)| {
            row.iter()
                .zip(input.iter())
                .map(|(w_ij, &in_j)| w_ij * in_j)
                .sum()
        })
        .collect::<Vec<f32>>()
}

// Partial Matrix Multiply
pub fn qk_mul(
    input: &[f32],
    w: &[Vec<f32>],
    idx_range: Range<usize>,
    tok_range: Range<usize>,
) -> Vec<f32> {
    tok_range
        .clone()
        .map(|i| {
            idx_range
                .clone()
                .into_iter()
                .map(|j| w[i][j] * input[j])
                .sum()
        })
        .collect::<Vec<f32>>()
}

pub fn qkv_mul(
    input: &[f32],
    w: &[Vec<f32>],
    idx_range: Range<usize>,
    tok_range: Range<usize>,
) -> Vec<f32> {
    idx_range
        .clone()
        .map(|i| tok_range.clone().map(|j| w[j][i] * input[j]).sum())
        .collect()
}

// --------------------------------------------------------------------------------
// Functions

pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn relu_vec(input: &[f32]) -> Vec<f32> {
    input.iter().map(|a: &f32| relu(*a)).collect::<Vec<_>>()
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn silu_vec(input: &[f32]) -> Vec<f32> {
    input.iter().map(|a: &f32| silu(*a)).collect::<Vec<_>>()
}

// --------------------------------------------------------------------------------
// Weighted RMS Normalization

pub fn rms_norm(input: &[f32], w: &[f32]) -> Vec<f32> {
    let sum: f32 = input.iter().map(|&x| x * x).sum();
    let norm = 1.0 / ((sum / input.len() as f32) + EPS).sqrt();
    input
        .iter()
        .zip(w.iter())
        .map(|(a, b)| *a * norm * *b)
        .collect()
}

// --------------------------------------------------------------------------------
// Softmax

pub fn softmax(input: &[f32]) -> Vec<f32> {
    let max_val = *input
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let exp_sum: f32 = input.iter().map(|&x| (x - max_val).exp()).sum();

    input
        .iter()
        .map(|input| (input - max_val).exp() / exp_sum)
        .collect::<Vec<_>>()
}

// --------------------------------------------------------------------------------
// Random Sampling

pub fn rand_sample(prob_dist: &[f32]) -> usize {
    let mut rng = rand::thread_rng();
    let rand: f32 = rng.gen();
    let mut cdf = 0.0;

    for (i, &prob) in prob_dist.iter().enumerate() {
        cdf += prob;
        if rand < cdf {
            return i;
        }
    }

    prob_dist.len() - 1
}

// --------------------------------------------------------------------------------
// RoPE: Positional Encoding

pub fn rope(
    q_out: &mut Vec<f32>,
    k_out: &mut Vec<f32>,
    q_in: &[f32],
    k_in: &[f32],
    cos_vec: &[f32],
    sin_vec: &[f32],
    head_begin: usize,
    head_dim: usize,
) {
    for i in 0..(head_dim / 2) {
        let i0 = head_begin + i * 2;
        let i1 = head_begin + i * 2 + 1;

        let q0 = q_in[i0];
        let q1 = q_in[i1];

        let k0 = k_in[i0];
        let k1 = k_in[i1];

        let cos = cos_vec[i];
        let sin = sin_vec[i];

        q_out[i0] = q0 * cos - q1 * sin;
        q_out[i1] = q0 * sin + q1 * cos;

        k_out[i0] = k0 * cos - k1 * sin;
        k_out[i1] = k0 * sin + k1 * cos;
    }
}
