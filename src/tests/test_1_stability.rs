#![allow(clippy::needless_range_loop)]
use crate::net::hopfield::Hopfield;
use crate::net::utility::{Dataset, OJA_EPOCHS, OJA_LR, OJA_TOL, Utils};

/// TEST 1: skuteczność reguł uczenia, liczba stabilnych wzorców i dlaczego
/// wzorce nie są stabilne (punkty 1 i 2 projektu)
pub struct StabilityTest;

impl StabilityTest {
    pub fn run(datasets: &[Dataset]) {
        Utils::print_header(
            "TEST 1: skuteczność reguł uczenia i stabilność wzorców",
            "Wzorzec uczący jest stabilny, jeśli podany na wejście sieci nie zmienia\n\
             się w żadnym neuronie (sign(W·x) = x). Hebb: pobudzenie neuronu =\n\
             sygnał (~1 - N/n) + przesłuch od pozostałych wzorców, proporcjonalny\n\
             do ich podobieństwa. Porównujemy Hebba i Oję na dostarczonych zbiorach.",
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

    /// 1a. cechy zbiorów i liczba stabilnych wzorców
    fn summary(datasets: &[Dataset], nets: &[(Hopfield, Hopfield)]) {
        println!("-- 1a. Podsumowanie (sieć uczona na całym zbiorze) --\n");
        println!(
            "{:<24}{:>6}{:>5}{:>7}{:>9}{:>9}{:>8}{:>8}{:>8}",
            "zbiór", "n", "N", "N/n", "aktywn.", "wspólne", "podob.", "Hebb", "Oja"
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
        println!(
            "\nN/n     = obciążenie sieci (pojemność teoretyczna dla losowych wzorców ~0.138)"
        );
        println!("aktywn. = średni udział czarnych pikseli (0.5 = zrównoważone)");
        println!("wspólne = % pikseli, które mają ten sam kolor we WSZYSTKICH wzorcach");
        println!("podob.  = średnie |x_p·x_q|/n po parach wzorców (0 = różne, 1 = identyczne)");
    }

    /// 1b. który wzorzec jest niestabilny, jak bardzo i czy ma podobnego "bliźniaka"
    fn per_pattern(datasets: &[Dataset], nets: &[(Hopfield, Hopfield)]) {
        println!("\n-- 1b. Szczegóły per wzorzec --");
        println!("nr        = numer wzorca");
        println!("Hebb, Oja = 'ok' albo ile pikseli zmienia się w pierwszym kroku");
        println!(
            "bliźn.    = numer najbardziej podobnego wzorca, podob. = podobieństwo do niego\n"
        );
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
            Utils::print_cells(&format!("  {:<7}", "nr"), &nr_cells);
            Utils::print_cells(&format!("  {:<7}", "Hebb"), &hebb_cells);
            Utils::print_cells(&format!("  {:<7}", "Oja"), &oja_cells);
            Utils::print_cells(&format!("  {:<7}", "bliźn."), &twin_cells);
            Utils::print_cells(&format!("  {:<7}", "podob."), &sim_cells);
        }
    }

    /// 1c. ile wzorców jest stabilnych przy uczeniu na pierwszych k
    fn cardinality(datasets: &[Dataset]) {
        println!("\n-- 1c. Wpływ liczby wzorców --");
        println!("Uczymy na pierwszych k wzorcach zbioru i liczymy, ile z tych k jest stabilnych.");
        println!("(dla dużych zbiorów Oja liczy się chwilę)\n");
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
