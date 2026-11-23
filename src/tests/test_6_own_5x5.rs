#![allow(clippy::needless_range_loop)]
use crate::net::hopfield::Hopfield;
use crate::net::utility::{OJA_EPOCHS, OJA_LR, OJA_TOL, Utils};

/// TEST 6: custom 5x5 patterns for stability and oscillation (project items 7 and 8)
pub struct Own5x5Test;

impl Own5x5Test {
    const STABLE_SET: &'static str = "src/own/stable-5x5.csv";
    const OSC_TRAIN: &'static str = "src/own/oscillation-train-5x5.csv";
    const OSC_INPUT: &'static str = "src/own/oscillation-input-5x5.csv";
    const OSC_NAMES: [&'static str; 2] = ["cross", "X"];

    pub fn run() {
        Utils::print_header(
            "TEST 6: custom 5x5 patterns",
            "6a: Find a large set of 25-element vectors that are all stable\n\
             under Hebbian learning.\n\
             6b: Choose a training set and an input that produce a two-state\n\
             cycle under synchronous updates.",
        );
        Self::stable_set();
        Self::oscillation();
    }

    /// 6a. Set of 19 nearly orthogonal vectors
    fn stable_set() {
        println!("-- 6a. Stable 5x5 pattern set --");
        println!(
            "Start with the 24 pairwise orthogonal rows of a Hadamard matrix\n\
             of order 24 (Paley construction). Append a 25th pixel of +1 so that\n\
             each pair has dot product 1. The margin is 26 - 2N > 0 for N <= 12.\n\
             Randomly searching subsets and row signs yielded a larger set.\n\
             A greedy search adding random vectors yielded 13 patterns.\n"
        );
        let (patterns, width, height) = Utils::load_patterns(Self::STABLE_SET);
        let k: usize = patterns.len();
        let n: usize = width * height;
        let hebb: Hopfield = Hopfield::new_hebb(&patterns);
        let oja: Hopfield = Hopfield::new_oja(&patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);
        println!(
            "set: {} patterns, N/n = {:.2} (capacity for random patterns ~0.138)",
            k,
            k as f32 / n as f32
        );
        println!(
            "min. Hebbian margin = {} (> 0 means all stable), max pairwise overlap = {:.3}",
            Self::hebb_min_margin(&patterns),
            Utils::most_similar_pair(&patterns).2
        );
        println!(
            "stable: Hebb {}/{}, Oja {}/{}\n",
            Utils::count_stable(&hebb, &patterns),
            k,
            Utils::count_stable(&oja, &patterns),
            k
        );
        Self::print_bitmaps(&patterns, width, height, 10);
    }

    /// 6b. Train on a cross and an X, then use a frame as input
    fn oscillation() {
        println!("-- 6b. Oscillation under synchronous updates --\n");
        let (training, width, height) = Utils::load_patterns(Self::OSC_TRAIN);
        let (inputs, _, _) = Utils::load_patterns(Self::OSC_INPUT);
        let input: &Vec<f32> = &inputs[0];
        let net: Hopfield = Hopfield::new_hebb(&training);

        println!(
            "Hebbian training set: cross, X. Stable: {}/{}",
            Utils::count_stable(&net, &training),
            training.len()
        );
        Self::print_bitmaps(&training, width, height, 10);
        println!("input: frame");
        Self::print_bitmaps(&inputs, width, height, 10);

        let mut probe: Vec<f32> = input.clone();
        let (history, end) = net.synchronized_test(&mut probe);
        println!("Synchronous updates. Result: {:?}. State history:", end);
        Self::print_bitmaps(&history, width, height, 10);

        let mut probe: Vec<f32> = input.clone();
        let history: Vec<Vec<f32>> = net.asynchronized_test(&mut probe);
        println!(
            "Asynchronous updates on the same input. Steps: {}. Final state: {}",
            history.len() - 1,
            Self::describe_state(&probe, &training)
        );
        Self::print_bitmaps(&[probe], width, height, 10);
    }

    /// HELPERS
    /// Minimal Hebb stability margin (without dividing by n):
    /// min over patterns q and neurons i of  x_i^q · z_i = Σ_p x_i^q x_i^p (x^p·x^q) − N
    fn hebb_min_margin(patterns: &[Vec<f32>]) -> f32 {
        let big_n: usize = patterns.len();
        let n: usize = patterns[0].len();
        let mut dots: Vec<Vec<f32>> = vec![vec![0.0; big_n]; big_n];
        for p in 0..big_n {
            for q in 0..big_n {
                for i in 0..n {
                    dots[p][q] += patterns[p][i] * patterns[q][i];
                }
            }
        }
        let mut min_margin: f32 = f32::MAX;
        for q in 0..big_n {
            for i in 0..n {
                let mut field: f32 = 0.0;
                for p in 0..big_n {
                    field += patterns[p][i] * dots[p][q];
                }
                min_margin = min_margin.min(patterns[q][i] * field - big_n as f32);
            }
        }
        min_margin
    }
    /// Which pattern / negative of which pattern / spurious state
    fn describe_state(state: &[f32], patterns: &[Vec<f32>]) -> String {
        for (pattern, name) in patterns.iter().zip(Self::OSC_NAMES.iter()) {
            if pattern.as_slice() == state {
                return name.to_string();
            }
            if pattern.iter().zip(state.iter()).all(|(a, b)| *a == -*b) {
                return format!("negative ({})", name);
            }
        }
        "spurious state".to_string()
    }
    /// Patterns as bitmaps side by side ('#' = black, '.' = white)
    fn print_bitmaps(patterns: &[Vec<f32>], width: usize, height: usize, per_row: usize) {
        for chunk in patterns.chunks(per_row) {
            for y in 0..height {
                print!("  ");
                for pattern in chunk {
                    for x in 0..width {
                        print!(
                            "{}",
                            if pattern[y * width + x] > 0.0 {
                                '#'
                            } else {
                                '.'
                            }
                        );
                    }
                    print!("   ");
                }
                println!();
            }
            println!();
        }
    }
}
