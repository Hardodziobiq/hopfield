use crate::net::hopfield::Hopfield;
use crate::net::utility::{Dataset, Mode, NOISE_REPEATS, OJA_EPOCHS, OJA_LR, OJA_TOL, SEED, Utils};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// TEST 4: zbiór z grupami podobnych wzorców vs zbiór zróżnicowany (punkt 5 projektu)
pub struct GroupsTest;

impl GroupsTest {
    const GROUP_SET: &'static str = "large-25x25.plus.csv";
    const GROUP_SIZES: [usize; 2] = [6, 8];
    const GROUP_NOISE: [f32; 3] = [0.1, 0.2, 0.3];
    const OUTCOME_NOISE: f32 = 0.2;

    pub fn run(datasets: &[Dataset]) {
        Utils::print_header(
            "TEST 4: zbiór z grupami podobnych wzorców vs zbiór zróżnicowany",
            "Z jednego pliku wybieramy dwa podzbiory tej samej liczności:\n\
             zróżnicowany (wzorce możliwie różne) i z grupami (pary najbardziej\n\
             podobnych wzorców). Ten sam rozmiar obrazków, to samo źródło.",
        );
        let ds: &Dataset = Utils::find_dataset(datasets, Self::GROUP_SET);
        let mut rng: StdRng = StdRng::seed_from_u64(SEED);
        println!("źródło: {} ({} wzorców)\n", ds.name, ds.patterns.len());

        for k in Self::GROUP_SIZES {
            println!(
                "==================== k = {} wzorców ====================\n",
                k
            );
            let prepared: Vec<(&str, Vec<Vec<f32>>, Hopfield, Hopfield)> = Self::chosen_sets(ds, k);
            Self::noise_recovery(&prepared, &mut rng);
            Self::outcomes(&prepared, &mut rng);
            println!();
        }
        println!("inny wz.  = sieć trafiła w INNY wzorzec ze zbioru (pomyłka)");
        println!("inny stan = stan, który nie jest żadnym wzorcem ani negatywem (pasożytniczy)");
    }

    /// 4a. wybór podzbiorów, ich podobieństwo i stabilność
    fn chosen_sets(
        ds: &Dataset,
        k: usize,
    ) -> Vec<(&'static str, Vec<Vec<f32>>, Hopfield, Hopfield)> {
        println!("-- 4a. Wybrane zbiory --\n");
        let variants: [(&'static str, Vec<usize>); 2] = [
            ("zróżnicowany", Self::diverse_subset(&ds.patterns, k)),
            ("z grupami", Self::grouped_subset(&ds.patterns, k)),
        ];
        let mut prepared: Vec<(&'static str, Vec<Vec<f32>>, Hopfield, Hopfield)> = Vec::new();
        for (name, idxs) in variants {
            let subset: Vec<Vec<f32>> = idxs.iter().map(|&i| ds.patterns[i].clone()).collect();
            let hebb: Hopfield = Hopfield::new_hebb(&subset);
            let oja: Hopfield = Hopfield::new_oja(&subset, OJA_LR, OJA_EPOCHS, OJA_TOL);
            println!("{}: wzorce {:?}", name, idxs);
            println!(
                "  podob. = {:.3}, max pod. = {:.3}, stabilne: Hebb {}/{}, Oja {}/{}",
                Utils::mean_similarity(&subset),
                Utils::most_similar_pair(&subset).2,
                Utils::count_stable(&hebb, &subset),
                k,
                Utils::count_stable(&oja, &subset),
                k
            );
            prepared.push((name, subset, hebb, oja));
        }
        println!("(z grupami: kolejne dwójki numerów to pary najbardziej podobnych wzorców)");
        prepared
    }

    /// 4b. odtwarzanie przy kilku poziomach szumu
    fn noise_recovery(prepared: &[(&str, Vec<Vec<f32>>, Hopfield, Hopfield)], rng: &mut StdRng) {
        println!("\n-- 4b. Odtwarzanie z szumu (async, odsetek dokładnie odtworzonych) --\n");
        print!("  {:<20}", "");
        for level in Self::GROUP_NOISE {
            print!("{:>7}", format!("{:.0}%", level * 100.0));
        }
        println!();
        for (name, subset, hebb, oja) in prepared {
            for (net_name, net) in [("Hebb", hebb), ("Oja", oja)] {
                print!("  {:<20}", format!("{} {}", name, net_name));
                for level in Self::GROUP_NOISE {
                    let damage = |p: &[f32], r: &mut StdRng| Utils::add_noise(p, level, r);
                    let (rate, _) = Utils::recovery_rate(net, subset, Mode::Async, rng, &damage);
                    print!("{:>7.2}", rate);
                }
                println!();
            }
        }
    }

    /// 4c. w jakim stanie kończy sieć
    fn outcomes(prepared: &[(&str, Vec<Vec<f32>>, Hopfield, Hopfield)], rng: &mut StdRng) {
        println!(
            "\n-- 4c. Gdzie ląduje sieć (szum {:.0}%, async, {} prób na wzorzec) --\n",
            Self::OUTCOME_NOISE * 100.0,
            NOISE_REPEATS
        );
        println!(
            "  {:<20}{:>10}{:>10}{:>10}{:>11}",
            "", "oryginał", "inny wz.", "negatyw", "inny stan"
        );
        for (name, subset, hebb, oja) in prepared {
            for (net_name, net) in [("Hebb", hebb), ("Oja", oja)] {
                let [orig, other, neg, spurious] = Self::outcome_counts(net, subset, rng);
                println!(
                    "  {:<20}{:>10}{:>10}{:>10}{:>11}",
                    format!("{} {}", name, net_name),
                    orig,
                    other,
                    neg,
                    spurious
                );
            }
        }
    }

    /// HELPERS (choosing subsets, classifying outcomes)
    /// Greedy: start from the least similar pair, then add the pattern with
    /// the smallest maximum similarity to the already chosen ones.
    fn diverse_subset(patterns: &[Vec<f32>], k: usize) -> Vec<usize> {
        let mut best: (usize, usize, f32) = (0, 1, f32::MAX);
        for p in 0..patterns.len() {
            for q in (p + 1)..patterns.len() {
                let sim: f32 = Utils::similarity(&patterns[p], &patterns[q]);
                if sim < best.2 {
                    best = (p, q, sim);
                }
            }
        }
        let mut chosen: Vec<usize> = vec![best.0, best.1];
        while chosen.len() < k {
            let mut candidate: (usize, f32) = (0, f32::MAX);
            for i in 0..patterns.len() {
                if chosen.contains(&i) {
                    continue;
                }
                let mut worst: f32 = 0.0;
                for &j in &chosen {
                    worst = worst.max(Utils::similarity(&patterns[i], &patterns[j]));
                }
                if worst < candidate.1 {
                    candidate = (i, worst);
                }
            }
            chosen.push(candidate.0);
        }
        chosen
    }
    /// Greedy: repeatedly take the most similar pair of not yet chosen patterns (k even).
    fn grouped_subset(patterns: &[Vec<f32>], k: usize) -> Vec<usize> {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < k {
            let mut best: (usize, usize, f32) = (0, 0, -1.0);
            for p in 0..patterns.len() {
                if chosen.contains(&p) {
                    continue;
                }
                for q in (p + 1)..patterns.len() {
                    if chosen.contains(&q) {
                        continue;
                    }
                    let sim: f32 = Utils::similarity(&patterns[p], &patterns[q]);
                    if sim > best.2 {
                        best = (p, q, sim);
                    }
                }
            }
            chosen.push(best.0);
            chosen.push(best.1);
        }
        chosen
    }
    /// [original, other pattern from the set, negative of a pattern, spurious state]
    fn outcome_counts(net: &Hopfield, patterns: &[Vec<f32>], rng: &mut StdRng) -> [usize; 4] {
        let mut counts: [usize; 4] = [0; 4];
        for _ in 0..NOISE_REPEATS {
            for original in patterns {
                let mut probe: Vec<f32> = Utils::add_noise(original, Self::OUTCOME_NOISE, rng);
                net.asynchronized_test(&mut probe);
                if probe == *original {
                    counts[0] += 1;
                } else if patterns.iter().any(|p| *p == probe) {
                    counts[1] += 1;
                } else if patterns
                    .iter()
                    .any(|p| p.iter().zip(probe.iter()).all(|(a, b)| *a == -*b))
                {
                    counts[2] += 1;
                } else {
                    counts[3] += 1;
                }
            }
        }
        counts
    }
}
