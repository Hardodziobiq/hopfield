use crate::net::hopfield::Hopfield;
use crate::net::utility::{Dataset, Mode, NOISE_REPEATS, OJA_EPOCHS, OJA_LR, OJA_TOL, SEED, Utils};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// TEST 3: odtwarzanie z uszkodzonego wejścia vs liczność i rodzaj zbioru (punkt 4 projektu)
pub struct RecoveryTest;

impl RecoveryTest {
    const CARD_NOISE: f32 = 0.2;
    const CARD_SETS: [&'static str; 2] = ["large-25x25.plus.csv", "letters-14x20.csv"];
    const OCCLUDE_FRACTION: f32 = 0.2;

    pub fn run(datasets: &[Dataset]) {
        Utils::print_header(
            "TEST 3: odtwarzanie oryginałów z uszkodzonego wejścia",
            "Jak skuteczność odtwarzania zależy od rodzaju zbioru (3a),\n\
             od liczby zapamiętanych wzorców (3b) i od rodzaju uszkodzenia (3c).",
        );
        let mut rng: StdRng = StdRng::seed_from_u64(SEED);
        Self::all_sets(datasets, &mut rng);
        Self::cardinality(datasets, &mut rng);
        Self::damage_type(datasets, &mut rng);
    }

    /// 3a. szum 0-50% na wszystkich zbiorach: dokładne odtworzenia i zgodne piksele
    fn all_sets(datasets: &[Dataset], rng: &mut StdRng) {
        println!("-- 3a. Odtwarzanie z szumu na wszystkich zbiorach --");
        println!("dokładne = odsetek prób zakończonych dokładnie oryginałem");
        println!("piksele  = średni udział pikseli zgodnych z oryginałem po odtworzeniu");
        println!("({} powtórzeń na wzorzec)\n", NOISE_REPEATS);
        for ds in datasets {
            let hebb: Hopfield = Hopfield::new_hebb(&ds.patterns);
            let oja: Hopfield = Hopfield::new_oja(&ds.patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);
            println!(
                "{} (N = {}, podob. = {:.3}):",
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
            println!("  dokładne:");
            Utils::print_levels_header();
            for (label, values) in &exact_rows {
                Utils::print_rate_row(label, values);
            }
            println!("  piksele:");
            Utils::print_levels_header();
            for (label, values) in &pixel_rows {
                Utils::print_rate_row(label, values);
            }
            println!();
        }
    }

    /// 3b. odtwarzanie przy uczeniu na pierwszych k wzorcach
    fn cardinality(datasets: &[Dataset], rng: &mut StdRng) {
        println!(
            "-- 3b. Wpływ liczności (szum {:.0}%, async) --",
            Self::CARD_NOISE * 100.0
        );
        println!("Uczymy na pierwszych k wzorcach, zaszumiamy każdy z nich i liczymy");
        println!("odsetek dokładnie odtworzonych. (Oja liczy się chwilę)\n");
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

    /// 3c. losowy szum vs zakrycie poziomym pasem tej samej wielkości
    fn damage_type(datasets: &[Dataset], rng: &mut StdRng) {
        println!(
            "\n-- 3c. Rodzaj uszkodzenia: szum vs zakrycie ({:.0}% obrazka, async) --",
            Self::OCCLUDE_FRACTION * 100.0
        );
        println!("szum     = losowe piksele odwrócone");
        println!("zakrycie = poziomy pas w losowym miejscu zamalowany na biało");
        println!("wartości = odsetek dokładnie odtworzonych\n");
        println!(
            "{:<24}{:>11}{:>11}{:>11}{:>11}",
            "zbiór", "Hebb szum", "Hebb zakr.", "Oja szum", "Oja zakr."
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
