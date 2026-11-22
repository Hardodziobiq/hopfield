#![allow(clippy::needless_range_loop)]
use crate::net::hopfield::Hopfield;
use crate::net::utility::{OJA_EPOCHS, OJA_LR, OJA_TOL, Utils};

/// TEST 6: własne zbiory 5x5 - stabilny zbiór i oscylacja (punkty 7 i 8 projektu)
pub struct Own5x5Test;

impl Own5x5Test {
    const STABLE_SET: &'static str = "src/own/stable-5x5.csv";
    const OSC_TRAIN: &'static str = "src/own/oscillation-train-5x5.csv";
    const OSC_INPUT: &'static str = "src/own/oscillation-input-5x5.csv";
    const OSC_NAMES: [&'static str; 2] = ["krzyż", "X"];

    pub fn run() {
        Utils::print_header(
            "TEST 6: własne zbiory 5x5",
            "6a: możliwie liczny zbiór wektorów długości 25, w którym wszystkie\n\
             wzorce są stabilne (reguła Hebba).\n\
             6b: zbiór uczący i wejście, które w trybie synchronicznym kończy się\n\
             oscylacją między dwoma stanami.",
        );
        Self::stable_set();
        Self::balanced_set();
        Self::oscillation();
    }

    /// The first 22 bits contain 11 ones; the final 3 bits are all equal.
    /// Symmetry gives two Hebbian margins for the entire family.
    fn balanced_set() {
        const BALANCED: u32 = 22;
        const HALF: u32 = BALANCED / 2;
        let count = (0..(1_u32 << BALANCED))
            .filter(|mask| mask.count_ones() == HALF)
            .count()
            * 2;
        let first_margin = count as f64 / (f64::from(BALANCED - 1) * 25.0);
        let last_margin = 2.0 * count as f64 / 25.0;
        println!("-- 6a. Konstrukcja z 22 zrównoważonymi pikselami i trzema równymi --");
        println!("Hebb: {count} różnych, ściśle stabilnych wzorców.");
        println!(
            "Margines pierwszych 22 pikseli: {first_margin:.3}, ostatnich trzech: {last_margin:.3}.\n"
        );
    }

    /// 6a. zbiór 19 prawie ortogonalnych wektorów
    fn stable_set() {
        println!("-- 6a. Stabilny zbiór 5x5 --");
        println!(
            "Wzorce wybrano z wierszy macierzy Hadamarda rzędu 24 z dodatkowym pikselem.\n\
             Poniżej najmniejszy margines x_i (Wx)_i dla całego zbioru.\n"
        );
        let (patterns, width, height) = Utils::load_patterns(Self::STABLE_SET);
        let k: usize = patterns.len();
        let n: usize = width * height;
        let hebb: Hopfield = Hopfield::new_hebb(&patterns);
        let oja: Hopfield = Hopfield::new_oja(&patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);
        println!(
            "zbiór: {} wzorców, N/n = {:.2} (pojemność dla losowych ~0.138)",
            k,
            k as f32 / n as f32
        );
        println!(
            "min. margines Hebba = {} (> 0 = wszystkie stabilne), max podobieństwo par = {:.3}",
            Self::hebb_min_margin(&patterns) / n as f32,
            Utils::most_similar_pair(&patterns).2
        );
        println!(
            "stabilne: Hebb {}/{}, Oja {}/{}\n",
            Utils::count_stable(&hebb, &patterns),
            k,
            Utils::count_stable(&oja, &patterns),
            k
        );
        Self::print_bitmaps(&patterns, width, height, 10);
    }

    /// 6b. krzyż + X, wejście: ramka
    fn oscillation() {
        println!("-- 6b. Oscylacja w trybie synchronicznym --\n");
        let (training, width, height) = Utils::load_patterns(Self::OSC_TRAIN);
        let (inputs, _, _) = Utils::load_patterns(Self::OSC_INPUT);
        let input: &Vec<f32> = &inputs[0];
        let net: Hopfield = Hopfield::new_hebb(&training);

        println!(
            "zbiór uczący (Hebb): krzyż, X, stabilne: {}/{}",
            Utils::count_stable(&net, &training),
            training.len()
        );
        Self::print_bitmaps(&training, width, height, 10);
        println!("wejście: ramka");
        Self::print_bitmaps(&inputs, width, height, 10);

        let mut probe: Vec<f32> = input.clone();
        let (history, end) = net.synchronized_test(&mut probe);
        println!("tryb synchroniczny, wynik: {:?}, kolejne stany:", end);
        Self::print_bitmaps(&history, width, height, 10);

        let mut probe: Vec<f32> = input.clone();
        let history: Vec<Vec<f32>> = net.asynchronized_test(&mut probe);
        println!(
            "tryb asynchroniczny dla tego samego wejścia: liczba kroków {}, stan końcowy: {}",
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
                return format!("negatyw ({})", name);
            }
        }
        "stan pasożytniczy".to_string()
    }
    /// Patterns as bitmaps side by side ('#' = black, '.' = white)
    fn print_bitmaps(patterns: &[Vec<f32>], width: usize, height: usize, per_row: usize) {
        for chunk in patterns.chunks(per_row) {
            for y in 0..height {
                print!("  ");
                for (index, pattern) in chunk.iter().enumerate() {
                    if index > 0 {
                        print!("   ");
                    }
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
                }
                println!();
            }
            println!();
        }
    }
}
