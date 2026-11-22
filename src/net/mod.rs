pub mod bitmap;
mod grouped_hopfield;
pub mod hopfield;
pub mod large_hopfield;
pub mod utility;

use utility::{Mode, OJA_EPOCHS, OJA_LR, OJA_TOL, TestEnd};

pub trait Recall {
    fn run(&self, probe: &mut Vec<f32>, mode: Mode) -> (Vec<Vec<f32>>, TestEnd);
}

impl Recall for hopfield::Hopfield {
    fn run(&self, probe: &mut Vec<f32>, mode: Mode) -> (Vec<Vec<f32>>, TestEnd) {
        match mode {
            Mode::Sync => self.synchronized_test(probe),
            Mode::Async => (self.asynchronized_test(probe), TestEnd::Stable),
        }
    }
}

impl Recall for large_hopfield::LargeHopfield {
    fn run(&self, probe: &mut Vec<f32>, mode: Mode) -> (Vec<Vec<f32>>, TestEnd) {
        match mode {
            Mode::Sync => self.synchronized_test(probe),
            Mode::Async => (self.asynchronized_test(probe), TestEnd::Stable),
        }
    }
}

/// Keep the original matrix for ordinary images and avoid a quadratic allocation
/// in the number of pixels for large images with few training patterns.
pub fn train_bitmap(patterns: &Vec<Vec<f32>>, oja: bool) -> Box<dyn Recall> {
    const MAX_DENSE_NEURONS: usize = 2_048;
    let large = patterns
        .first()
        .is_some_and(|pattern| pattern.len() > MAX_DENSE_NEURONS);
    match (large, oja) {
        (false, false) => Box::new(hopfield::Hopfield::new_hebb(patterns)),
        (false, true) => Box::new(hopfield::Hopfield::new_oja(
            patterns, OJA_LR, OJA_EPOCHS, OJA_TOL,
        )),
        (true, false) => Box::new(large_hopfield::LargeHopfield::new(patterns.clone())),
        (true, true) => Box::new(grouped_hopfield::GroupedHopfield::new_oja(
            patterns, OJA_LR, OJA_EPOCHS, OJA_TOL,
        )),
    }
}
