use std::path::Path;

use crate::net::bitmap::Bitmaps;
use crate::net::hopfield::Hopfield;
use crate::net::utility::{Mode, OJA_EPOCHS, OJA_LR, OJA_TOL, SEED, Utils};
use rand::SeedableRng;
use rand::rngs::StdRng;

pub fn run() -> Result<(), String> {
    let data = Bitmaps::load(Path::new("src/large"))?;
    let patterns = &data.patterns;
    let hebb = Hopfield::new_hebb(patterns);
    let oja = Hopfield::new_oja(patterns, OJA_LR, OJA_EPOCHS, OJA_TOL);

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
    for (name, net) in [("Hebb", &hebb), ("Oja", &oja)] {
        println!(
            "{name}: stabilne {}/{}",
            Utils::count_stable(net, patterns),
            patterns.len()
        );
        for (label, mode) in [("sync", Mode::Sync), ("async", Mode::Async)] {
            let mut rng = StdRng::seed_from_u64(SEED);
            let (exact, pixels) = Utils::noise_test_stats(net, patterns, mode, &mut rng);
            Utils::print_levels_header();
            Utils::print_rate_row(&format!("{label} dokładne"), &exact);
            Utils::print_rate_row(&format!("{label} piksele"), &pixels);
        }
    }
    Ok(())
}
