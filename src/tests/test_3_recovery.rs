use crate::net::hopfield::Hopfield;
use crate::net::utility::{Dataset, Mode, NOISE_REPEATS, OJA_EPOCHS, OJA_LR, OJA_TOL, SEED, Utils};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// TEST 3: recovery from damaged inputs by dataset size and type (project item 4)
pub struct RecoveryTest;

impl RecoveryTest {
    const CARD_NOISE: f32 = 0.2;
    const CARD_SETS: [&'static str; 2] = ["large-25x25.plus.csv", "letters-14x20.csv"];
    const OCCLUDE_FRACTION: f32 = 0.2;

    pub fn run(datasets: &[Dataset]) {
        Utils::print_header(
            "TEST 3: recovery from damaged inputs",
            "We measure how recovery depends on dataset type (3a),\n\
             the number of learned patterns (3b), and the type of damage (3c).",
        );
        let mut rng: StdRng = StdRng::seed_from_u64(SEED);
        Self::all_sets(datasets, &mut rng);
        Self::cardinality(datasets, &mut rng);
        Self::damage_type(datasets, &mut rng);
    }

    /// 3a. Measure exact recovery and pixel agreement at 0–50% noise across all datasets
    fn all_sets(datasets: &[Dataset], rng: &mut StdRng) {
        println!("-- 3a. Recovery from noise across all datasets --");
        println!("exact  = fraction of trials that recovered the original pattern");
        println!("pixels = mean fraction of pixels matching the original after recovery");
        println!("({} trials per pattern)\n", NOISE_REPEATS);
        for ds in datasets {
            let hebb: Hopfield = Hopfield::new_hebb(&ds.patterns);
            let oja: Hopfield = Hopfield::new_oja(&ds.patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);
            println!(
                "{} (N = {}, overlap = {:.3}):",
                ds.name,
                ds.patterns.len(),
                Utils::mean_similarity(&ds.patterns)
            );
            let mut exact_rows: Vec<(String, Vec<f32>)> = Vec::new();
            let mut pixel_rows: Vec<(String, Vec<f32>)> = Vec::new();
            for (net_name, net) in [("Hebb", &hebb), ("Oja", &oja)] {
                for (mode_name, mode) in [("sync", Mode::Sync), ("async", Mode::Async)] {
                    let (exact, pixels) = Utils::noise_test_stats(net, &ds.patterns, mode, rng);
                    let label: String = format!("{} {}", net_name, mode_name);
                    exact_rows.push((label.clone(), exact));
                    pixel_rows.push((label, pixels));
                }
            }
            println!("  exact:");
            Utils::print_levels_header();
            for (label, values) in &exact_rows {
                Utils::print_rate_row(label, values);
            }
            println!("  pixels:");
            Utils::print_levels_header();
            for (label, values) in &pixel_rows {
                Utils::print_rate_row(label, values);
            }
            println!();
        }
    }

    /// 3b. Measure recovery after training on the first k patterns
    fn cardinality(datasets: &[Dataset], rng: &mut StdRng) {
        println!(
            "-- 3b. Effect of training set size ({:.0}% noise, async) --",
            Self::CARD_NOISE * 100.0
        );
        println!("Train on the first k patterns, add noise to each, and measure");
        println!("the fraction recovered exactly. (Oja training may take a moment.)\n");
        let damage = |p: &[f32], r: &mut StdRng| Utils::add_noise(p, Self::CARD_NOISE, r);
        for name in Self::CARD_SETS {
            let ds: &Dataset = Utils::find_dataset(datasets, name);
            let mut ks: Vec<String> = Vec::new();
            let mut hebb_cells: Vec<String> = Vec::new();
            let mut oja_cells: Vec<String> = Vec::new();
            for k in 1..=ds.patterns.len() {
                let subset: Vec<Vec<f32>> = ds.patterns[..k].to_vec();
                let hebb: Hopfield = Hopfield::new_hebb(&subset);
                let oja: Hopfield = Hopfield::new_oja(&subset, OJA_LR, OJA_EPOCHS, OJA_TOL);
                let (hebb_rate, _) =
                    Utils::recovery_rate(&hebb, &subset, Mode::Async, rng, &damage);
                let (oja_rate, _) = Utils::recovery_rate(&oja, &subset, Mode::Async, rng, &damage);
                ks.push(k.to_string());
                hebb_cells.push(format!("{:.2}", hebb_rate));
                oja_cells.push(format!("{:.2}", oja_rate));
            }
            println!("{}:", ds.name);
            Utils::print_cells(&format!("  {:<5}", "k"), &ks);
            Utils::print_cells(&format!("  {:<5}", "Hebb"), &hebb_cells);
            Utils::print_cells(&format!("  {:<5}", "Oja"), &oja_cells);
        }
    }

    /// 3c. Compare random pixel flips with a horizontal band covering the same fraction of the image
    fn damage_type(datasets: &[Dataset], rng: &mut StdRng) {
        println!(
            "\n-- 3c. Damage type: noise vs occlusion ({:.0}% of the image, async) --",
            Self::OCCLUDE_FRACTION * 100.0
        );
        println!("noise     = random pixel flips");
        println!("occlusion = a white horizontal band at a random position");
        println!("values    = fraction recovered exactly\n");
        println!(
            "{:<24}{:>11}{:>11}{:>11}{:>11}",
            "dataset", "Hebb noise", "Hebb band", "Oja noise", "Oja band"
        );
        println!("{}", "-".repeat(68));
        for ds in datasets {
            let hebb: Hopfield = Hopfield::new_hebb(&ds.patterns);
            let oja: Hopfield = Hopfield::new_oja(&ds.patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);
            let band_h: usize =
                ((Self::OCCLUDE_FRACTION * ds.height as f32).round() as usize).max(1);
            let noise = |p: &[f32], r: &mut StdRng| Utils::add_noise(p, Self::OCCLUDE_FRACTION, r);
            let occlusion = |p: &[f32], r: &mut StdRng| {
                let y0: usize = r.random_range(0..=(ds.height - band_h));
                Utils::occlude(p, ds.width, ds.height, 0, y0, ds.width, band_h, -1.0)
            };
            let (hn, _) = Utils::recovery_rate(&hebb, &ds.patterns, Mode::Async, rng, &noise);
            let (ho, _) = Utils::recovery_rate(&hebb, &ds.patterns, Mode::Async, rng, &occlusion);
            let (on, _) = Utils::recovery_rate(&oja, &ds.patterns, Mode::Async, rng, &noise);
            let (oo, _) = Utils::recovery_rate(&oja, &ds.patterns, Mode::Async, rng, &occlusion);
            println!(
                "{:<24}{:>11.2}{:>11.2}{:>11.2}{:>11.2}",
                ds.name, hn, ho, on, oo
            );
        }
    }
}
