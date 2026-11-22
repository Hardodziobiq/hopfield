#![allow(clippy::needless_range_loop)]
use crate::net::hopfield::Hopfield;
use crate::net::utility::{Dataset, OJA_EPOCHS, OJA_LR, OJA_TOL, Utils};

/// TEST 1: learning rule performance, how many patterns are stable and why
/// some patterns are unstable (project items 1 and 2)
pub struct StabilityTest;

impl StabilityTest {
    pub fn run(datasets: &[Dataset]) {
        Utils::print_header(
            "TEST 1: learning rule performance and pattern stability",
            "A training pattern is stable if the network leaves every neuron\n\
             unchanged (sign(W·x) = x). Under Hebb, the local field combines\n\
             the signal (~1 - N/n) and crosstalk proportional to pattern similarity.\n\
             We compare Hebb and Oja on the supplied datasets.",
        );
        let mut nets: Vec<(Hopfield, Hopfield)> = Vec::new();
        for ds in datasets {
            let hebb: Hopfield = Hopfield::new_hebb(&ds.patterns);
            let oja: Hopfield = Hopfield::new_oja(&ds.patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);
            nets.push((hebb, oja));
        }
        Self::summary(datasets, &nets);
        Self::per_pattern(datasets, &nets);
        Self::cardinality(datasets);
    }

    /// 1a. dataset properties and number of stable patterns
    fn summary(datasets: &[Dataset], nets: &[(Hopfield, Hopfield)]) {
        println!("-- 1a. Summary (trained on the full dataset) --\n");
        println!(
            "{:<24}{:>6}{:>5}{:>7}{:>9}{:>9}{:>8}{:>8}{:>8}",
            "dataset", "n", "N", "N/n", "activity", "shared", "overlap", "Hebb", "Oja"
        );
        println!("{}", "-".repeat(84));
        for (ds, (hebb, oja)) in datasets.iter().zip(nets.iter()) {
            let n: usize = ds.width * ds.height;
            let big_n: usize = ds.patterns.len();
            println!(
                "{:<24}{:>6}{:>5}{:>7.3}{:>9.3}{:>8.1}%{:>8.3}{:>8}{:>8}",
                ds.name,
                n,
                big_n,
                big_n as f32 / n as f32,
                Self::activity(&ds.patterns),
                100.0 * Self::common_pixels(&ds.patterns),
                Utils::mean_similarity(&ds.patterns),
                format!("{}/{}", Utils::count_stable(hebb, &ds.patterns), big_n),
                format!("{}/{}", Utils::count_stable(oja, &ds.patterns), big_n),
            );
        }
        println!("\nN/n     = network load (theoretical capacity for random patterns ~0.138)");
        println!("activity = mean fraction of black pixels (0.5 means balanced)");
        println!("shared  = percentage of pixels with the same color in every pattern");
        println!("overlap = mean |x_p·x_q|/n across pairs (1 for identical or opposite patterns)");
    }

    /// 1b. Which patterns are unstable, how many pixels flip, and which pattern is closest
    fn per_pattern(datasets: &[Dataset], nets: &[(Hopfield, Hopfield)]) {
        println!("\n-- 1b. Per-pattern details --");
        println!("index     = pattern index");
        println!("Hebb, Oja = 'ok' or the number of pixels that flip in the first step");
        println!("closest   = index of the most similar pattern, overlap = similarity to it\n");
        for (ds, (hebb, oja)) in datasets.iter().zip(nets.iter()) {
            let mut nr_cells: Vec<String> = Vec::new();
            let mut hebb_cells: Vec<String> = Vec::new();
            let mut oja_cells: Vec<String> = Vec::new();
            let mut twin_cells: Vec<String> = Vec::new();
            let mut sim_cells: Vec<String> = Vec::new();
            for (idx, pattern) in ds.patterns.iter().enumerate() {
                let (twin, sim) = Self::twin_of(&ds.patterns, idx);
                nr_cells.push(idx.to_string());
                hebb_cells.push(Self::flips_cell(Utils::unstable_pixels(hebb, pattern)));
                oja_cells.push(Self::flips_cell(Utils::unstable_pixels(oja, pattern)));
                twin_cells.push(twin.to_string());
                sim_cells.push(format!("{:.2}", sim));
            }
            println!("{}:", ds.name);
            Utils::print_cells(&format!("  {:<7}", "index"), &nr_cells);
            Utils::print_cells(&format!("  {:<7}", "Hebb"), &hebb_cells);
            Utils::print_cells(&format!("  {:<7}", "Oja"), &oja_cells);
            Utils::print_cells(&format!("  {:<7}", "closest"), &twin_cells);
            Utils::print_cells(&format!("  {:<7}", "overlap"), &sim_cells);
        }
    }

    /// 1c. Number of stable patterns after training on the first k
    fn cardinality(datasets: &[Dataset]) {
        println!("\n-- 1c. Effect of training set size --");
        println!("Train on the first k patterns and count how many of them are stable.");
        println!("(Oja training may take a moment on large datasets)\n");
        for ds in datasets {
            let mut ks: Vec<String> = Vec::new();
            let mut hebb_cells: Vec<String> = Vec::new();
            let mut oja_cells: Vec<String> = Vec::new();
            for k in 1..=ds.patterns.len() {
                let subset: Vec<Vec<f32>> = ds.patterns[..k].to_vec();
                let hebb: Hopfield = Hopfield::new_hebb(&subset);
                let oja: Hopfield = Hopfield::new_oja(&subset, OJA_LR, OJA_EPOCHS, OJA_TOL);
                ks.push(k.to_string());
                hebb_cells.push(Utils::count_stable(&hebb, &subset).to_string());
                oja_cells.push(Utils::count_stable(&oja, &subset).to_string());
            }
            println!("{}:", ds.name);
            Utils::print_cells(&format!("  {:<5}", "k"), &ks);
            Utils::print_cells(&format!("  {:<5}", "Hebb"), &hebb_cells);
            Utils::print_cells(&format!("  {:<5}", "Oja"), &oja_cells);
        }
    }

    /// HELPERS (pattern statistics used only here)
    fn flips_cell(flips: usize) -> String {
        if flips == 0 {
            "ok".to_string()
        } else {
            flips.to_string()
        }
    }
    fn activity(patterns: &[Vec<f32>]) -> f32 {
        let mut black: usize = 0;
        let mut total: usize = 0;
        for pattern in patterns {
            for &value in pattern {
                if value > 0.0 {
                    black += 1;
                }
                total += 1;
            }
        }
        black as f32 / total as f32
    }
    fn common_pixels(patterns: &[Vec<f32>]) -> f32 {
        let n: usize = patterns[0].len();
        let mut common: usize = 0;
        for i in 0..n {
            if patterns.iter().all(|p| p[i] == patterns[0][i]) {
                common += 1;
            }
        }
        common as f32 / n as f32
    }
    /// (index, similarity) of the pattern most similar to patterns[idx]
    fn twin_of(patterns: &[Vec<f32>], idx: usize) -> (usize, f32) {
        let mut best: (usize, f32) = (idx, 0.0);
        for q in 0..patterns.len() {
            if q == idx {
                continue;
            }
            let sim: f32 = Utils::similarity(&patterns[idx], &patterns[q]);
            if sim > best.1 {
                best = (q, sim);
            }
        }
        best
    }
}
