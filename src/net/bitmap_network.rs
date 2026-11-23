use super::grouped_hopfield;
use super::hopfield::Hopfield;
use super::large_hopfield::LargeHopfield;
use super::utility::{Mode, OJA_EPOCHS, OJA_LR, OJA_TOL, TestEnd};

pub trait Recall {
    fn run(&self, probe: &mut Vec<f32>, mode: Mode) -> (Vec<Vec<f32>>, TestEnd);
}

impl Recall for Hopfield {
    fn run(&self, probe: &mut Vec<f32>, mode: Mode) -> (Vec<Vec<f32>>, TestEnd) {
        match mode {
            Mode::Sync => self.synchronized_test(probe),
            Mode::Async => (self.asynchronized_test(probe), TestEnd::Stable),
        }
    }
}

impl Recall for LargeHopfield {
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
        (false, false) => Box::new(Hopfield::new_hebb(patterns)),
        (false, true) => Box::new(Hopfield::new_oja(patterns, OJA_LR, OJA_EPOCHS, OJA_TOL)),
        (true, false) => Box::new(LargeHopfield::new(patterns.clone())),
        (true, true) => Box::new(grouped_hopfield::GroupedHopfield::new_oja(
            patterns, OJA_LR, OJA_EPOCHS, OJA_TOL,
        )),
    }
}
