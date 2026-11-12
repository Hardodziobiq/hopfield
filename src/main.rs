use std::fs;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use hopfield::net::hopfield::Hopfield;
use hopfield::net::large_hopfield::LargeHopfield;
use hopfield::net::utility::{TestEnd, Utils};

const DATA_DIR: &str = "src/data";
const SEED: u64 = 42;

const OJA_LR: f32 = 0.5;
const OJA_EPOCHS: usize = 300;
const OJA_TOL: f32 = 1e-6;

const NOISE_LEVELS: [f32; 6] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
const NOISE_REPEATS: usize = 10;
const RANDOM_PROBES: usize = 100;

#[derive(Clone, Copy)]
enum Mode {
    Sync,
    Async,
}

fn main() {
    let mut rng: StdRng = StdRng::seed_from_u64(SEED);

    for path in csv_files(DATA_DIR) {
        let (patterns, width, height) = Utils::load_patterns(&path);
        let n: usize = width * height;

        println!(
            "\n===== {} ({}x{}, n = {}, wzorców = {}) =====",
            path,
            width,
            height,
            n,
            patterns.len()
        );
        println!(
            "obciążenie N/n: {:.3} (pojemność teoretyczna ~0.138)",
            patterns.len() as f32 / n as f32
        );
        println!("średnie podobieństwo |x_p·x_q|/n: {:.3}", mean_similarity(&patterns));

        let hebb: Hopfield = Hopfield::new_hebb(&patterns);
        let oja: Hopfield = Hopfield::new_oja(&patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);
        let nets: [(&str, &Hopfield); 2] = [("Hebb", &hebb), ("Oja", &oja)];
        let modes: [(&str, Mode); 2] = [("sync", Mode::Sync), ("async", Mode::Async)];

        // A. STABILNOŚĆ WZORCÓW
        println!("-- stabilne wzorce --");
        for (name, net) in nets {
            println!("{:6} {}/{}", name, count_stable(net, &patterns), patterns.len());
        }

        // C. ODTWARZANIE Z SZUMU
        println!("-- odtwarzanie z szumu (odsetek idealnie odtworzonych) --");
        print!("{:12}", "");
        for level in NOISE_LEVELS {
            print!("{:>7}", format!("{:.0}%", level * 100.0));
        }
        println!();
        for (name, net) in nets {
            for (mode_name, mode) in modes {
                let rates: Vec<f32> = noise_test(net, &patterns, mode, &mut rng);
                print!("{:12}", format!("{} {}", name, mode_name));
                for rate in rates {
                    print!("{:>7.2}", rate);
                }
                println!();
            }
        }

        // E. LOSOWE WEJŚCIE
        println!(
            "-- losowe wejście ({} prób): wzorzec / negatyw / pasożytniczy / cykl --",
            RANDOM_PROBES
        );
        for (name, net) in nets {
            for (mode_name, mode) in modes {
                let [pattern, negative, spurious, cycle] =
                    random_test(net, &patterns, n, mode, &mut rng);
                println!(
                    "{:12}{:>5} {:>5} {:>5} {:>5}",
                    format!("{} {}", name, mode_name),
                    pattern,
                    negative,
                    spurious,
                    cycle
                );
            }
        }
    }

    // SPRAWDZENIE: LargeHopfield musi dawać to samo co zwykły Hebb
    let (patterns, _, _) = Utils::load_patterns(&format!("{}/large-25x25.csv", DATA_DIR));
    let (same, total) = large_equivalence(&patterns, &mut rng);
    println!(
        "\nLargeHopfield vs Hopfield (Hebb) na large-25x25: {}/{} identycznych przebiegów",
        same, total
    );
}

/// Wszystkie pliki .csv z katalogu, posortowane po nazwie.
fn csv_files(dir: &str) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    for entry in fs::read_dir(dir).expect("Cannot read data dir!") {
        let path: String = entry.expect("Bad dir entry!").path().to_string_lossy().to_string();
        if path.ends_with(".csv") {
            files.push(path);
        }
    }
    files.sort();
    files
}

/// Uruchamia odtwarzanie w wybranym trybie. Async zawsze kończy się stanem stabilnym.
fn recall(net: &Hopfield, probe: &mut Vec<f32>, mode: Mode) -> TestEnd {
    match mode {
        Mode::Sync => net.synchronized_test(probe).1,
        Mode::Async => {
            net.asynchronized_test(probe);
            TestEnd::Stable
        }
    }
}

/// Ile wzorców jest stanami stabilnymi (żaden neuron nie zmienia się w pierwszym kroku).
fn count_stable(net: &Hopfield, patterns: &[Vec<f32>]) -> usize {
    let mut count: usize = 0;
    for pattern in patterns {
        let mut probe: Vec<f32> = pattern.clone();
        let (history, _) = net.synchronized_test(&mut probe);
        if history.len() == 1 {
            count += 1;
        }
    }
    count
}

/// Średnie |x_p · x_q| / n po wszystkich parach wzorców (0 = różne, 1 = identyczne lub negatywy).
fn mean_similarity(patterns: &[Vec<f32>]) -> f32 {
    let n: f32 = patterns[0].len() as f32;
    let mut sum: f32 = 0.0;
    let mut pairs: usize = 0;
    for p in 0..patterns.len() {
        for q in (p + 1)..patterns.len() {
            let mut dot: f32 = 0.0;
            for i in 0..patterns[p].len() {
                dot += patterns[p][i] * patterns[q][i];
            }
            sum += (dot / n).abs();
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        sum / pairs as f32
    }
}

/// Dla każdego poziomu szumu: odsetek prób, w których sieć odtworzyła dokładnie oryginał.
fn noise_test(net: &Hopfield, patterns: &[Vec<f32>], mode: Mode, rng: &mut StdRng) -> Vec<f32> {
    let mut rates: Vec<f32> = Vec::new();
    for level in NOISE_LEVELS {
        let mut ok: usize = 0;
        let mut total: usize = 0;
        for _ in 0..NOISE_REPEATS {
            for original in patterns {
                let mut probe: Vec<f32> = Utils::add_noise(original, level, rng);
                let end: TestEnd = recall(net, &mut probe, mode);
                if end == TestEnd::Stable && probe == *original {
                    ok += 1;
                }
                total += 1;
            }
        }
        rates.push(ok as f32 / total as f32);
    }
    rates
}

/// Losowe wejścia: [trafił we wzorzec, w negatyw wzorca, stan pasożytniczy, cykl].
fn random_test(
    net: &Hopfield,
    patterns: &[Vec<f32>],
    n: usize,
    mode: Mode,
    rng: &mut StdRng,
) -> [usize; 4] {
    let mut counts: [usize; 4] = [0; 4];
    for _ in 0..RANDOM_PROBES {
        let mut probe: Vec<f32> = Vec::new();
        for _ in 0..n {
            probe.push(if rng.random_bool(0.5) { 1.0 } else { -1.0 });
        }
        let end: TestEnd = recall(net, &mut probe, mode);
        if end == TestEnd::Cycle {
            counts[3] += 1;
        } else if patterns.iter().any(|p| *p == probe) {
            counts[0] += 1;
        } else if patterns
            .iter()
            .any(|p| p.iter().zip(probe.iter()).all(|(a, b)| *a == -*b))
        {
            counts[1] += 1;
        } else {
            counts[2] += 1;
        }
    }
    counts
}

/// Porównuje przebiegi Hopfield (Hebb) i LargeHopfield dla tych samych zaszumionych wejść.
fn large_equivalence(patterns: &Vec<Vec<f32>>, rng: &mut StdRng) -> (usize, usize) {
    let net: Hopfield = Hopfield::new_hebb(patterns);
    let large: LargeHopfield = LargeHopfield::new(patterns.clone());
    let mut same: usize = 0;
    let mut total: usize = 0;
    for original in patterns {
        let probe: Vec<f32> = Utils::add_noise(original, 0.2, rng);

        let mut a: Vec<f32> = probe.clone();
        let mut b: Vec<f32> = probe.clone();
        if net.synchronized_test(&mut a) == large.synchronized_test(&mut b) {
            same += 1;
        }
        total += 1;

        let mut a: Vec<f32> = probe.clone();
        let mut b: Vec<f32> = probe;
        if net.asynchronized_test(&mut a) == large.asynchronized_test(&mut b) {
            same += 1;
        }
        total += 1;
    }
    (same, total)
}