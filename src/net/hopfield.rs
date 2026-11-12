#![allow(clippy::needless_range_loop)]
use crate::net::utility::{TestEnd, Utils};

/// HOPFIELD STRUCT
pub struct Hopfield {
    n: usize,
    weights: Vec<Vec<f32>>,
}

/// HOPFIELD MAIN FUNCTIONS
impl Hopfield {
    pub fn new_hebb(patterns: &Vec<Vec<f32>>) -> Self {
        let n: usize = Utils::pattern_validate(patterns);
        Self {
            n,
            weights: Self::weights_hebb(patterns, n),
        }
    }
    pub fn new_oja(
        patterns: &Vec<Vec<f32>>,
        learning_rate: f32,
        epoch_count: usize,
        tol: f32,
    ) -> Self {
        let n: usize = Utils::pattern_validate(patterns);
        Self {
            n,
            weights: Self::weights_oja(patterns, n, learning_rate, epoch_count, tol),
        }
    }
    pub fn synchronized_test(&self, probe: &mut Vec<f32>) -> (Vec<Vec<f32>>, TestEnd) {
        self.test_validate(probe);
        self.sync_test(probe)
    }
    pub fn asynchronized_test(&self, probe: &mut Vec<f32>) -> Vec<Vec<f32>> {
        self.test_validate(probe);
        self.async_test(probe)
    }
}

/// HOPFIELD UTILITY FUNCTIONS
impl Hopfield {
    /// VALIDATION
    fn test_validate(&self, probe: &[f32]) {
        if self.n != probe.len() {
            panic!("Given test pattern length doesn't match!");
        }
    }

    /// TRAINING UTILS
    fn weights_hebb(patterns: &Vec<Vec<f32>>, n: usize) -> Vec<Vec<f32>> {
        let mut weights: Vec<Vec<f32>> = vec![vec![0.0; n]; n];
        let n_f32: f32 = n as f32;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                for pattern in patterns {
                    weights[i][j] += pattern[i] * pattern[j];
                }
                weights[i][j] /= n_f32;
            }
        }
        weights
    }
    fn weights_oja(
        patterns: &Vec<Vec<f32>>,
        n: usize,
        learning_rate: f32,
        epoch_count: usize,
        tol: f32,
    ) -> Vec<Vec<f32>> {
        let mut weights: Vec<Vec<f32>> = Self::weights_hebb(patterns, n);
        for _ in 0..epoch_count {
            let mut max_change: f32 = 0.0;
            for pattern in patterns {
                let y: Vec<f32> = Self::matrix_times_vector(&weights, pattern);
                let v: Vec<f32> = Self::matrix_t_times_vector(&weights, &y);
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let d: f32 = learning_rate * y[i] * (pattern[j] - v[j]);
                        if !d.is_finite() {
                            panic!("Oja: weights diverged (NaN/inf), decrease learning_rate!");
                        }
                        weights[i][j] += d;
                        max_change = max_change.max(d.abs());
                    }
                }
            }
            Self::sym(&mut weights, n);
            if max_change < tol {
                break;
            }
        }
        weights
    }
    fn matrix_times_vector(matrix: &[Vec<f32>], vec: &[f32]) -> Vec<f32> {
        let mut new_vec: Vec<f32> = vec![0.0; vec.len()];
        for (row_idx, row_values) in matrix.iter().enumerate() {
            for (col_idx, value) in row_values.iter().enumerate() {
                new_vec[row_idx] += *value * vec[col_idx];
            }
        }
        new_vec
    }
    fn matrix_t_times_vector(matrix: &[Vec<f32>], vec: &[f32]) -> Vec<f32> {
        let mut new_vec: Vec<f32> = vec![0.0; vec.len()];
        for (row_idx, row_values) in matrix.iter().enumerate() {
            for (col_idx, value) in row_values.iter().enumerate() {
                new_vec[col_idx] += *value * vec[row_idx];
            }
        }
        new_vec
    }
    fn sym(matrix: &mut [Vec<f32>], n: usize) {
        for i in 0..n {
            for j in (i + 1)..n {
                if matrix[i][j] != matrix[j][i] {
                    let val: f32 = (matrix[i][j] + matrix[j][i]) / 2.0;
                    matrix[i][j] = val;
                    matrix[j][i] = val;
                }
            }
        }
    }

    /// SYNC & ASYNC TESTS
    fn sync_test(&self, probe: &mut Vec<f32>) -> (Vec<Vec<f32>>, TestEnd) {
        let mut history: Vec<Vec<f32>> = vec![probe.clone()];
        let mut last_probe: Vec<f32>;
        let mut before_last: Vec<f32> = Vec::new();
        let mut value: f32;
        loop {
            last_probe = probe.clone();
            for (row_idx, cond) in probe.iter_mut().enumerate() {
                value = 0.0;
                for (idx, weight) in self.weights[row_idx].iter().enumerate() {
                    if row_idx == idx {
                        continue;
                    }
                    value += weight * last_probe[idx];
                }
                *cond = Utils::sign(value, *cond);
            }
            if *probe == last_probe {
                return (history, TestEnd::Stable);
            }
            history.push(probe.clone());
            if *probe == before_last {
                return (history, TestEnd::Cycle);
            }
            before_last = last_probe;
        }
    }
    fn async_test(&self, probe: &mut Vec<f32>) -> Vec<Vec<f32>> {
        let mut history: Vec<Vec<f32>> = vec![probe.clone()];
        let mut last_probe: Vec<f32>;
        let mut value: f32;
        loop {
            last_probe = probe.clone();
            for row_idx in 0..probe.len() {
                value = 0.0;
                for (idx, weight) in self.weights[row_idx].iter().enumerate() {
                    if row_idx == idx {
                        continue;
                    }
                    value += weight * probe[idx];
                }
                probe[row_idx] = Utils::sign(value, probe[row_idx]);
            }
            if *probe == last_probe {
                return history;
            }
            history.push(probe.clone());
        }
    }
}
