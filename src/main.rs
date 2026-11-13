use hopfield::net::utility::{Dataset, Utils};
use hopfield::tests::test_1_stability::StabilityTest;
use hopfield::tests::test_2_common_block::CommonBlockTest;
use hopfield::tests::test_3_recovery::RecoveryTest;
use hopfield::tests::test_4_groups::GroupsTest;
use hopfield::tests::test_5_random_input::RandomInputTest;
use hopfield::tests::test_6_own_5x5::Own5x5Test;

const DATA_DIR: &str = "src/data";

/// cargo run --release -- test_number_1 test_number_2 etc.
fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let selected = |nr: usize| args.is_empty() || args.contains(&nr);
    let datasets: Vec<Dataset> = Utils::load_datasets(DATA_DIR);

    if selected(1) {
        StabilityTest::run(&datasets);
    }
    if selected(2) {
        CommonBlockTest::run(&datasets);
    }
    if selected(3) {
        RecoveryTest::run(&datasets);
    }
    if selected(4) {
        GroupsTest::run(&datasets);
    }
    if selected(5) {
        RandomInputTest::run(&datasets);
    }
    if selected(6) {
        Own5x5Test::run();
    }
}
