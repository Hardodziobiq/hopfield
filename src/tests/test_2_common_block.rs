#![allow(clippy::needless_range_loop)]
use crate::net::hopfield::Hopfield;
use crate::net::utility::{Dataset, Mode, OJA_EPOCHS, OJA_LR, OJA_TOL, SEED, Utils};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// TEST 2: nearly identical 25x25 and 25x50 patterns (project item 3)
pub struct CommonBlockTest;

impl CommonBlockTest {
    const SMALL_SET: &'static str = "large-25x25.csv";
    const BIG_SET: &'static str = "large-25x50.csv";

    pub fn run(datasets: &[Dataset]) {
        Utils::print_header(
            "TEST 2: nearly identical 25x25 and 25x50 patterns",
            "We compare the two datasets and measure how their differences affect\n\
             pattern similarity and recovery from noise. The 25x50 control uses\n\
             a different random bottom block for each pattern.",
        );
        let small: &Dataset = Utils::find_dataset(datasets, Self::SMALL_SET);
        let big: &Dataset = Utils::find_dataset(datasets, Self::BIG_SET);
        let mut rng: StdRng = StdRng::seed_from_u64(SEED);

        Self::structure(small, big);
        Self::pair_overlaps(small, big);

        let control: Vec<Vec<f32>> = Self::random_block_set(small, big, &mut rng);
        let variants: [(&str, &Vec<Vec<f32>>); 3] = [
            ("25x25", &small.patterns),
            ("25x50 shared block", &big.patterns),
            ("25x50 random block", &control),
        ];
        let nets: Vec<(Hopfield, Hopfield)> = Self::stability(&variants);
        Self::noise_recovery(&variants, &nets, &mut rng);
    }

    /// 2a. Compare the top half of 25x50 with the 25x25 pattern and inspect the bottom half
    fn structure(small: &Dataset, big: &Dataset) {
        let n_small: usize = small.width * small.height;
        let n_big: usize = big.width * big.height;
        println!("-- 2a. Structure of the 25x50 patterns --");
        println!("top diff.    = top-half pixels that differ from the 25x25 pattern");
        println!("bottom black = percentage of black pixels in the bottom half\n");
        println!("{:<10}{:>14}{:>15}", "pattern", "top diff.", "bottom black");
        println!("{}", "-".repeat(39));
        for idx in 0..small.patterns.len() {
            let ps: &Vec<f32> = &small.patterns[idx];
            let pb: &Vec<f32> = &big.patterns[idx];
            let diff: usize = (0..n_small).filter(|&i| ps[i] != pb[i]).count();
            let black: usize = (n_small..n_big).filter(|&i| pb[i] > 0.0).count();
            println!(
                "{:<10}{:>14}{:>14.1}%",
                idx,
                diff,
                100.0 * black as f32 / (n_big - n_small) as f32
            );
        }
    }

    /// 2b. Compare observed pairwise overlap with the prediction (o + 1) / 2
    fn pair_overlaps(small: &Dataset, big: &Dataset) {
        println!("\n-- 2b. Observed and predicted pairwise overlap --");
        println!("o = x_p·x_q / n retains the sign (negative values indicate opposing patterns)");
        println!("With a shared bottom block, o_25x50 = (o_25x25 + 1) / 2\n");
        println!(
            "{:<8}{:>11}{:>11}{:>13}",
            "pair", "o 25x25", "o 25x50", "predicted"
        );
        println!("{}", "-".repeat(43));
        for p in 0..small.patterns.len() {
            for q in (p + 1)..small.patterns.len() {
                let o_small: f32 = Self::overlap(&small.patterns[p], &small.patterns[q]);
                let o_big: f32 = Self::overlap(&big.patterns[p], &big.patterns[q]);
                println!(
                    "{:<8}{:>11.3}{:>11.3}{:>13.3}",
                    format!("{}-{}", p, q),
                    o_small,
                    o_big,
                    (o_small + 1.0) / 2.0
                );
            }
        }
    }

    /// 2c. Measure stability for each variant and return the networks for 2d
    fn stability(variants: &[(&str, &Vec<Vec<f32>>); 3]) -> Vec<(Hopfield, Hopfield)> {
        println!("\n-- 2c. Pattern stability --\n");
        println!(
            "{:<22}{:>6}{:>9}{:>10}{:>10}",
            "variant", "n", "overlap", "Hebb", "Oja"
        );
        println!("{}", "-".repeat(57));
        let mut nets: Vec<(Hopfield, Hopfield)> = Vec::new();
        for (name, patterns) in variants {
            let hebb: Hopfield = Hopfield::new_hebb(patterns);
            let oja: Hopfield = Hopfield::new_oja(patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);
            println!(
                "{:<22}{:>6}{:>9.3}{:>10}{:>10}",
                name,
                patterns[0].len(),
                Utils::mean_similarity(patterns),
                format!(
                    "{}/{}",
                    Utils::count_stable(&hebb, patterns),
                    patterns.len()
                ),
                format!("{}/{}", Utils::count_stable(&oja, patterns), patterns.len()),
            );
            nets.push((hebb, oja));
        }
        nets
    }

    /// 2d. Measure recovery from noise for each variant
    fn noise_recovery(
        variants: &[(&str, &Vec<Vec<f32>>); 3],
        nets: &[(Hopfield, Hopfield)],
        rng: &mut StdRng,
    ) {
        println!("\n-- 2d. Recovery from noise (fraction recovered exactly) --\n");
        for ((name, patterns), (hebb, oja)) in variants.iter().zip(nets.iter()) {
            println!("{}:", name);
            Utils::print_levels_header();
            for (net_name, net) in [("Hebb", hebb), ("Oja", oja)] {
                for (mode_name, mode) in [("sync", Mode::Sync), ("async", Mode::Async)] {
                    let (exact, _) = Utils::noise_test_stats(net, patterns, mode, rng);
                    Utils::print_rate_row(&format!("{} {}", net_name, mode_name), &exact);
                }
            }
        }
    }

    /// HELPERS
    /// x_p · x_q / n WITH sign (unlike Utils::similarity)
    fn overlap(a: &[f32], b: &[f32]) -> f32 {
        let mut dot: f32 = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
        }
        dot / a.len() as f32
    }
    /// 25x25 patterns + random bottom block, different for every pattern
    fn random_block_set(small: &Dataset, big: &Dataset, rng: &mut StdRng) -> Vec<Vec<f32>> {
        let extra: usize = big.width * big.height - small.width * small.height;
        let mut control: Vec<Vec<f32>> = Vec::new();
        for pattern in &small.patterns {
            let mut extended: Vec<f32> = pattern.clone();
            for _ in 0..extra {
                extended.push(if rng.random_bool(0.5) { 1.0 } else { -1.0 });
            }
            control.push(extended);
        }
        control
    }
}
