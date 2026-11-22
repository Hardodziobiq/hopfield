#![allow(clippy::needless_range_loop)]
use crate::net::hopfield::Hopfield;
use crate::net::utility::{Dataset, Mode, OJA_EPOCHS, OJA_LR, OJA_TOL, SEED, Utils};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// TEST 2: prawie identyczne wzorce 25x25 vs 25x50 (punkt 3 projektu)
pub struct CommonBlockTest;

impl CommonBlockTest {
    const SMALL_SET: &'static str = "large-25x25.csv";
    const BIG_SET: &'static str = "large-25x50.csv";

    pub fn run(datasets: &[Dataset]) {
        Utils::print_header(
            "TEST 2: prawie identyczne wzorce 25x25 vs 25x50",
            "Sprawdzamy, czym różnią się oba zbiory, jak to wpływa na podobieństwo\n\
             wzorców i na odtwarzanie z szumu. Eksperyment kontrolny: 25x50, w którym\n\
             dolny blok jest losowy i inny dla każdego wzorca (zamiast wspólnego).",
        );
        let small: &Dataset = Utils::find_dataset(datasets, Self::SMALL_SET);
        let big: &Dataset = Utils::find_dataset(datasets, Self::BIG_SET);
        let mut rng: StdRng = StdRng::seed_from_u64(SEED);

        Self::structure(small, big);
        Self::pair_overlaps(small, big);

        let control: Vec<Vec<f32>> = Self::random_block_set(small, big, &mut rng);
        let variants: [(&str, &Vec<Vec<f32>>); 3] = [
            ("25x25", &small.patterns),
            ("25x50 wspólny blok", &big.patterns),
            ("25x50 losowy blok", &control),
        ];
        let nets: Vec<(Hopfield, Hopfield)> = Self::stability(&variants);
        Self::noise_recovery(&variants, &nets, &mut rng);
    }

    /// 2a. górna połowa 25x50 vs wzorzec 25x25, dolna połowa
    fn structure(small: &Dataset, big: &Dataset) {
        let n_small: usize = small.width * small.height;
        let n_big: usize = big.width * big.height;
        println!("-- 2a. Struktura wzorców 25x50 --");
        println!("górna różn.  = ile pikseli górnej połowy różni się od wzorca 25x25");
        println!("dolna czarne = % czarnych pikseli w dolnej połowie\n");
        println!(
            "{:<10}{:>14}{:>15}",
            "wzorzec", "górna różn.", "dolna czarne"
        );
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

    /// 2b. podobieństwo każdej pary: zmierzone vs przewidziane (o + 1) / 2
    fn pair_overlaps(small: &Dataset, big: &Dataset) {
        println!("\n-- 2b. Podobieństwo par: zmierzone vs przewidziane --");
        println!("o = x_p·x_q / n ze znakiem (ujemne = wzorce 'przeciwne')");
        println!("jeśli dolny blok jest wspólny: o_25x50 = (o_25x25 + 1) / 2\n");
        println!(
            "{:<8}{:>11}{:>11}{:>13}",
            "para", "o 25x25", "o 25x50", "przewidyw."
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

    /// 2c. stabilność dla trzech wariantów (sieci zwracamy do 2d)
    fn stability(variants: &[(&str, &Vec<Vec<f32>>); 3]) -> Vec<(Hopfield, Hopfield)> {
        println!("\n-- 2c. Stabilność wzorców --\n");
        println!(
            "{:<22}{:>6}{:>9}{:>10}{:>10}",
            "wariant", "n", "podob.", "Hebb", "Oja"
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

    /// 2d. odtwarzanie z szumu dla trzech wariantów
    fn noise_recovery(
        variants: &[(&str, &Vec<Vec<f32>>); 3],
        nets: &[(Hopfield, Hopfield)],
        rng: &mut StdRng,
    ) {
        println!("\n-- 2d. Odtwarzanie z szumu (odsetek dokładnie odtworzonych) --\n");
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
