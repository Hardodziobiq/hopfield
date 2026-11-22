use std::collections::HashMap;

use crate::net::utility::{Mode, TestEnd};

const FIELD_TOLERANCE: f64 = 1e-10;

/// Neurons whose values agree in every training pattern share a weight row.
/// Complementary training columns also share a row, after a sign change.
pub struct GroupedHopfield {
    groups: Vec<usize>,
    signs: Vec<f32>,
    weights: Vec<Vec<f64>>,
}

impl GroupedHopfield {
    #[cfg(test)]
    fn new_hebb(patterns: &[Vec<f32>]) -> Self {
        Self::train(patterns, None)
    }

    pub fn new_oja(patterns: &[Vec<f32>], learning_rate: f32, epochs: usize, tol: f32) -> Self {
        assert!(learning_rate.is_finite() && learning_rate >= 0.0);
        assert!(tol.is_finite() && tol >= 0.0);
        Self::train(patterns, Some((learning_rate, epochs, tol)))
    }

    fn train(patterns: &[Vec<f32>], oja: Option<(f32, usize, f32)>) -> Self {
        let n = patterns.first().map_or(0, Vec::len);
        assert!(
            n > 0
                && patterns.iter().all(|pattern| {
                    pattern.len() == n && pattern.iter().all(|&pixel| pixel == -1.0 || pixel == 1.0)
                }),
            "Training requires nonempty bipolar patterns of equal length"
        );
        let signs = patterns[0].clone();
        let mut signatures = Vec::<Vec<i8>>::new();
        let mut signature_indices = HashMap::new();
        let mut counts = Vec::<usize>::new();
        let mut groups = Vec::with_capacity(n);
        for (pixel, &sign) in signs.iter().enumerate() {
            let signature: Vec<i8> = patterns.iter().map(|p| (p[pixel] * sign) as i8).collect();
            let group = *signature_indices
                .entry(signature.clone())
                .or_insert_with(|| {
                    signatures.push(signature);
                    counts.push(0);
                    counts.len() - 1
                });
            groups.push(group);
            counts[group] += 1;
        }

        let rows = counts.len();
        let mut weights = vec![vec![0.0; rows]; rows];
        for (i, a) in signatures.iter().enumerate() {
            for j in 0..=i {
                if i == j && counts[i] == 1 {
                    continue;
                }
                let dot: i64 = a
                    .iter()
                    .zip(&signatures[j])
                    .map(|(&a, &b)| i64::from(a * b))
                    .sum();
                let weight = dot as f64 / n as f64;
                weights[i][j] = weight;
                weights[j][i] = weight;
            }
        }
        if let Some((rate, epochs, tol)) = oja {
            let examples: Vec<Vec<f64>> = (0..patterns.len())
                .map(|p| signatures.iter().map(|s| f64::from(s[p])).collect())
                .collect();
            Self::learn_oja(
                &mut weights,
                &counts,
                &examples,
                f64::from(rate) / n as f64,
                epochs,
                f64::from(tol),
            );
        }

        Self {
            groups,
            signs,
            weights,
        }
    }

    fn learn_oja(
        weights: &mut [Vec<f64>],
        counts: &[usize],
        examples: &[Vec<f64>],
        eta: f64,
        epochs: usize,
        tol: f64,
    ) {
        let rows = weights.len();
        let scale = weights
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .zip(counts)
                    .map(|(w, &count)| w.abs() * count as f64)
                    .sum::<f64>()
                    - row[i].abs()
            })
            .fold(0.0, f64::max);
        if scale > 0.0 {
            for weight in weights.iter_mut().flatten() {
                *weight /= scale;
            }
        }

        let mut y = vec![0.0; rows];
        let mut v = vec![0.0; rows];
        for _ in 0..epochs {
            let mut max_change: f64 = 0.0;
            for x in examples {
                for (i, row) in weights.iter().enumerate() {
                    y[i] = row
                        .iter()
                        .zip(x)
                        .zip(counts)
                        .map(|((&weight, &input), &count)| weight * input * count as f64)
                        .sum::<f64>()
                        - row[i] * x[i];
                }
                for i in 0..rows {
                    v[i] = weights
                        .iter()
                        .zip(&y)
                        .zip(counts)
                        .map(|((row, &output), &count)| row[i] * output * count as f64)
                        .sum::<f64>()
                        - weights[i][i] * y[i];
                }
                for (i, row) in weights.iter_mut().enumerate() {
                    for (j, weight) in row.iter_mut().enumerate() {
                        if i == j && counts[i] == 1 {
                            continue;
                        }
                        let delta = eta * y[i] * (x[j] - v[j]);
                        *weight += delta;
                        assert!(weight.is_finite(), "Oja: decrease the learning rate");
                        max_change = max_change.max(delta.abs());
                    }
                }
            }
            for i in 1..weights.len() {
                let (previous, current) = weights.split_at_mut(i);
                let row = &mut current[0];
                for (j, other_row) in previous.iter_mut().enumerate() {
                    let mean = row[j] * 0.5 + other_row[i] * 0.5;
                    row[j] = mean;
                    other_row[i] = mean;
                }
            }
            if max_change < tol {
                break;
            }
        }
    }

    fn validate_probe(&self, probe: &[f32]) {
        assert_eq!(
            probe.len(),
            self.groups.len(),
            "Probe length must match the network"
        );
        assert!(
            probe.iter().all(|&x| x == -1.0 || x == 1.0),
            "Probe must be bipolar"
        );
    }

    /// One synchronous step, or one ordered sweep with immediate updates.
    #[cfg(test)]
    fn step(&self, probe: &mut [f32], mode: Mode) -> bool {
        self.validate_probe(probe);
        self.sweep(probe, mode)
    }

    fn sweep(&self, probe: &mut [f32], mode: Mode) -> bool {
        let mut sums = vec![0.0; self.weights.len()];
        for (i, &value) in probe.iter().enumerate() {
            sums[self.groups[i]] += f64::from(value * self.signs[i]);
        }
        let mut fields: Vec<f64> = self
            .weights
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&sums)
                    .map(|(weight, sum)| weight * sum)
                    .sum()
            })
            .collect();
        let mut changed = false;
        for (i, value) in probe.iter_mut().enumerate() {
            let group = self.groups[i];
            let old = *value * self.signs[i];
            // weights[group][group] is the coupling of distinct neurons in
            // the group. Subtract this neuron's own contribution.
            let field = fields[group] - self.weights[group][group] * f64::from(old);
            if field * f64::from(old) < -FIELD_TOLERANCE {
                *value = -*value;
                changed = true;
                if mode == Mode::Async {
                    let delta = -2.0 * f64::from(old);
                    for (field, row) in fields.iter_mut().zip(&self.weights) {
                        *field += row[group] * delta;
                    }
                }
            }
        }
        changed
    }

    pub fn run(&self, probe: &mut [f32], mode: Mode) -> (Vec<Vec<f32>>, TestEnd) {
        self.validate_probe(probe);
        let mut history = vec![probe.to_vec()];
        while self.sweep(probe, mode) {
            let cycle =
                mode == Mode::Sync && history.len() >= 2 && probe == history[history.len() - 2];
            history.push(probe.to_vec());
            if cycle {
                return (history, TestEnd::Cycle);
            }
        }
        (history, TestEnd::Stable)
    }
}

impl super::Recall for GroupedHopfield {
    fn run(&self, probe: &mut Vec<f32>, mode: Mode) -> (Vec<Vec<f32>>, TestEnd) {
        self.run(probe.as_mut_slice(), mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_training_and_steps_match_full_matrix() {
        // Includes equal and opposite columns, plus single-neuron groups.
        let patterns = vec![
            vec![1., 1., -1., 1., 1., 1.],
            vec![1., 1., -1., -1., 1., -1.],
            vec![-1., -1., 1., -1., 1., 1.],
        ];
        let n = patterns[0].len();
        for use_oja in [false, true] {
            let mut weights = vec![vec![0.0; n]; n];
            for (i, row) in weights.iter_mut().enumerate() {
                for (j, weight) in row.iter_mut().enumerate().filter(|(j, _)| *j != i) {
                    *weight =
                        patterns.iter().map(|x| f64::from(x[i] * x[j])).sum::<f64>() / n as f64;
                }
            }
            let net = if use_oja {
                let scale = weights
                    .iter()
                    .map(|row| row.iter().map(|w| w.abs()).sum::<f64>())
                    .fold(0.0, f64::max);
                for weight in weights.iter_mut().flatten() {
                    *weight /= scale;
                }
                for _ in 0..8 {
                    for x in &patterns {
                        let y: Vec<f64> = weights
                            .iter()
                            .map(|row| row.iter().zip(x).map(|(&w, &x)| w * f64::from(x)).sum())
                            .collect();
                        let v: Vec<f64> = (0..n)
                            .map(|j| weights.iter().zip(&y).map(|(row, &y)| row[j] * y).sum())
                            .collect();
                        for (i, row) in weights.iter_mut().enumerate() {
                            for (j, weight) in row.iter_mut().enumerate().filter(|(j, _)| *j != i) {
                                *weight += 0.5 / n as f64 * y[i] * (f64::from(x[j]) - v[j]);
                            }
                        }
                    }
                    let before = weights.clone();
                    for (i, row) in weights.iter_mut().enumerate() {
                        for (j, weight) in row.iter_mut().enumerate() {
                            *weight = (before[i][j] + before[j][i]) / 2.0;
                        }
                    }
                }
                GroupedHopfield::new_oja(&patterns, 0.5, 8, 0.0)
            } else {
                GroupedHopfield::new_hebb(&patterns)
            };
            for (i, row) in weights.iter().enumerate() {
                for (j, &expected) in row.iter().enumerate().filter(|(j, _)| *j != i) {
                    let actual = net.weights[net.groups[i]][net.groups[j]]
                        * f64::from(net.signs[i] * net.signs[j]);
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            for mask in 0..(1 << n) {
                let input: Vec<f32> = (0..n)
                    .map(|i| if mask & (1 << i) == 0 { -1.0 } else { 1.0 })
                    .collect();
                for mode in [Mode::Sync, Mode::Async] {
                    let mut expected = input.clone();
                    let mut actual = input.clone();
                    for (i, row) in weights.iter().enumerate() {
                        let state = if mode == Mode::Sync {
                            &input
                        } else {
                            &expected
                        };
                        let field = row
                            .iter()
                            .zip(state)
                            .map(|(&w, &s)| w * f64::from(s))
                            .sum::<f64>();
                        if field.abs() > FIELD_TOLERANCE {
                            expected[i] = field.signum() as f32;
                        }
                    }
                    assert_eq!(net.step(&mut actual, mode), expected != input);
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn synchronous_cycle_and_asynchronous_fixed_point() {
        let net = GroupedHopfield::new_hebb(&[vec![1.0, 1.0]]);
        let input = vec![1.0, -1.0];
        let (history, end) = net.run(&mut input.clone(), Mode::Sync);
        assert_eq!(end, TestEnd::Cycle);
        assert_eq!(history, [input.clone(), vec![-1.0, 1.0], input.clone()]);
        let (history, end) = net.run(&mut input.clone(), Mode::Async);
        assert_eq!(end, TestEnd::Stable);
        assert_eq!(history, [input, vec![-1.0, -1.0]]);
    }
}
