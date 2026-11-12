/// UTILS FUNCTIONS
pub struct Utils;

/// UTILS MAIN FUNCTIONS
impl Utils {
    pub fn pattern_validate(patterns: &Vec<Vec<f32>>) -> usize {
        if patterns.is_empty() {
            panic!("Empty patterns!");
        }
        if patterns[0].is_empty() {
            panic!("First pattern empty!");
        }
        let pattern_len: usize = patterns[0].len();
        for pattern in patterns {
            if pattern.len() != pattern_len {
                panic!("Different patterns length!");
            }
        }
        pattern_len
    }
    pub fn sign(value: f32, old: f32) -> f32 {
        if 0.0 < value {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            old
        }
    }
}

/// TEST END ENUM
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestEnd {
    Stable,
    Cycle,
}
