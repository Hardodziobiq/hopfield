use crate::net::hopfield::Hopfield;
use crate::net::utility::{Dataset, Mode, OJA_EPOCHS, OJA_LR, OJA_TOL, SEED, TestEnd, Utils};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// TEST 5: reakcja nauczonej sieci na losowe wejście (punkt 6 projektu)
pub struct RandomInputTest;

/// Wyniki dla jednej sieci i jednego trybu
struct RandomStats {
    counts: [usize; 4],        // wzorzec, negatyw, pasożytniczy, cykl
    distinct: usize,           // ile różnych stanów końcowych
    mean_steps: f32,           // średnia liczba kroków
    spurious_sim: Option<f32>, // średnie podobieństwo stanu pasożytniczego do najbliższego wzorca
    hits: Vec<usize>,          // ile razy trafiono w każdy wzorzec (lub jego negatyw)
}

impl RandomInputTest {
    const RANDOM_PROBES: usize = 100;

    pub fn run(datasets: &[Dataset]) {
        Utils::print_header(
            "TEST 5: reakcja nauczonej sieci na losowe wejście",
            "Podajemy losowe wektory ±1 i sprawdzamy, w jakim stanie sieć się\n\
             zatrzymuje, ile różnych atraktorów odwiedza i jak duże są baseny\n\
             przyciągania poszczególnych wzorców.",
        );
        let mut rng: StdRng = StdRng::seed_from_u64(SEED);
        println!(
            "-- 5a. Gdzie ląduje sieć ({} losowych wejść) --",
            Self::RANDOM_PROBES
        );
        println!("różnych   = liczba różnych stanów końcowych");
        println!("kroki     = średnia liczba kroków do zatrzymania");
        println!("pod. pas. = średnie podobieństwo stanu pasożytniczego do najbliższego wzorca");
        println!("5b        = ile razy sieć trafiła w dany wzorzec (async, wzorzec lub negatyw)\n");

        for ds in datasets {
            let hebb: Hopfield = Hopfield::new_hebb(&ds.patterns);
            let oja: Hopfield = Hopfield::new_oja(&ds.patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);
            println!("{} (N = {}):", ds.name, ds.patterns.len());
            println!(
                "  {:<12}{:>9}{:>9}{:>8}{:>6}{:>9}{:>7}{:>11}",
                "", "wzorzec", "negatyw", "pasoż.", "cykl", "różnych", "kroki", "pod. pas."
            );
            let mut hit_rows: Vec<(&str, Vec<usize>)> = Vec::new();
            for (net_name, net) in [("Hebb", &hebb), ("Oja", &oja)] {
                for (mode_name, mode) in [("sync", Mode::Sync), ("async", Mode::Async)] {
                    let stats: RandomStats = Self::random_stats(net, &ds.patterns, mode, &mut rng);
                    let sim_text: String = match stats.spurious_sim {
                        Some(sim) => format!("{:.2}", sim),
                        None => "-".to_string(),
                    };
                    println!(
                        "  {:<12}{:>9}{:>9}{:>8}{:>6}{:>9}{:>7.1}{:>11}",
                        format!("{} {}", net_name, mode_name),
                        stats.counts[0],
                        stats.counts[1],
                        stats.counts[2],
                        stats.counts[3],
                        stats.distinct,
                        stats.mean_steps,
                        sim_text
                    );
                    if mode == Mode::Async {
                        hit_rows.push((net_name, stats.hits));
                    }
                }
            }
            println!("  5b. baseny przyciągania:");
            let nr_cells: Vec<String> = (0..ds.patterns.len()).map(|i| i.to_string()).collect();
            Utils::print_cells(&format!("    {:<5}", "nr"), &nr_cells);
            for (net_name, hits) in &hit_rows {
                let cells: Vec<String> = hits.iter().map(|h| h.to_string()).collect();
                Utils::print_cells(&format!("    {:<5}", net_name), &cells);
            }
            println!();
        }
    }

    /// HELPERS
    /// Feeds RANDOM_PROBES random inputs and collects statistics.
    fn random_stats(
        net: &Hopfield,
        patterns: &[Vec<f32>],
        mode: Mode,
        rng: &mut StdRng,
    ) -> RandomStats {
        let n: usize = patterns[0].len();
        let mut counts: [usize; 4] = [0; 4];
        let mut finals: Vec<Vec<f32>> = Vec::new();
        let mut steps_sum: usize = 0;
        let mut sim_sum: f32 = 0.0;
        let mut sim_count: usize = 0;
        let mut hits: Vec<usize> = vec![0; patterns.len()];

        for _ in 0..Self::RANDOM_PROBES {
            let mut probe: Vec<f32> = Vec::new();
            for _ in 0..n {
                probe.push(if rng.random_bool(0.5) { 1.0 } else { -1.0 });
            }
            let (steps, end) = match mode {
                Mode::Sync => {
                    let (history, end) = net.synchronized_test(&mut probe);
                    (history.len() - 1, end)
                }
                Mode::Async => {
                    let history: Vec<Vec<f32>> = net.asynchronized_test(&mut probe);
                    (history.len() - 1, TestEnd::Stable)
                }
            };
            steps_sum += steps;
            if !finals.contains(&probe) {
                finals.push(probe.clone());
            }
            if end == TestEnd::Cycle {
                counts[3] += 1;
                continue;
            }
            let mut matched: bool = false;
            for (idx, pattern) in patterns.iter().enumerate() {
                if *pattern == probe {
                    counts[0] += 1;
                    hits[idx] += 1;
                    matched = true;
                    break;
                }
                if pattern.iter().zip(probe.iter()).all(|(a, b)| *a == -*b) {
                    counts[1] += 1;
                    hits[idx] += 1;
                    matched = true;
                    break;
                }
            }
            if !matched {
                counts[2] += 1;
                let mut best: f32 = 0.0;
                for pattern in patterns {
                    best = best.max(Utils::similarity(pattern, &probe));
                }
                sim_sum += best;
                sim_count += 1;
            }
        }
        RandomStats {
            counts,
            distinct: finals.len(),
            mean_steps: steps_sum as f32 / Self::RANDOM_PROBES as f32,
            spurious_sim: if sim_count > 0 {
                Some(sim_sum / sim_count as f32)
            } else {
                None
            },
            hits,
        }
    }
}
