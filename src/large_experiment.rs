use std::path::Path;

use hopfield::net::bitmap::Bitmaps;
use hopfield::net::bitmap_network::{Recall, train_bitmap};
use hopfield::net::utility::{Mode, NOISE_LEVELS, NOISE_REPEATS, SEED, TestEnd, Utils};
use rand::SeedableRng;
use rand::rngs::StdRng;

pub fn run() -> Result<(), String> {
    let data = Bitmaps::load(Path::new("src/large"))?;
    let patterns = &data.patterns;

    Utils::print_header(
        "TEST 7. Duże bitmapy",
        "Trzy zdjęcia po binaryzacji. Odzysk dokładny oraz zgodność pikseli.",
    );
    println!(
        "{}×{}, {} neuronów, {} wzorce, średni |overlap| {:.3}",
        data.width,
        data.height,
        data.width * data.height,
        patterns.len(),
        Utils::mean_similarity(patterns)
    );
    for (name, oja) in [("Hebb", false), ("Oja", true)] {
        let net = train_bitmap(patterns, oja);
        let stable = patterns
            .iter()
            .filter(|pattern| {
                let mut probe = (*pattern).clone();
                net.run(&mut probe, Mode::Sync).0.len() == 1
            })
            .count();
        println!("{name}: stabilne {stable}/{}", patterns.len());
        for (label, mode) in [("sync", Mode::Sync), ("async", Mode::Async)] {
            let mut rng = StdRng::seed_from_u64(SEED);
            let (exact, pixels): (Vec<_>, Vec<_>) = NOISE_LEVELS
                .iter()
                .map(|&noise| measure(net.as_ref(), patterns, mode, noise, &mut rng))
                .unzip();
            Utils::print_levels_header();
            Utils::print_rate_row(&format!("{label} dokładne"), &exact);
            Utils::print_rate_row(&format!("{label} piksele"), &pixels);
        }
    }
    Ok(())
}

fn measure(
    net: &dyn Recall,
    patterns: &[Vec<f32>],
    mode: Mode,
    noise: f32,
    rng: &mut StdRng,
) -> (f32, f32) {
    let mut exact = 0;
    let mut similarity = 0.0;
    for _ in 0..NOISE_REPEATS {
        for original in patterns {
            let mut probe = Utils::add_noise(original, noise, rng);
            let (_, end) = net.run(&mut probe, mode);
            exact += usize::from(end == TestEnd::Stable && probe == *original);
            let equal = probe.iter().zip(original).filter(|(a, b)| a == b).count();
            similarity += equal as f32 / original.len() as f32;
        }
    }
    let trials = (NOISE_REPEATS * patterns.len()) as f32;
    (exact as f32 / trials, similarity / trials)
}
