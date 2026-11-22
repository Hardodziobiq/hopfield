use std::path::Path;

use crate::net::bitmap::Bitmaps;
use crate::net::train_bitmap;
use crate::net::utility::{Mode, SEED, Utils};
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
        println!(
            "{name}: stabilne {}/{}",
            Utils::count_stable(net.as_ref(), patterns),
            patterns.len()
        );
        for (label, mode) in [("sync", Mode::Sync), ("async", Mode::Async)] {
            let mut rng = StdRng::seed_from_u64(SEED);
            let (exact, pixels) = Utils::noise_test_stats(net.as_ref(), patterns, mode, &mut rng);
            Utils::print_levels_header();
            Utils::print_rate_row(&format!("{label} dokładne"), &exact);
            Utils::print_rate_row(&format!("{label} piksele"), &pixels);
        }
    }
    Ok(())
}
