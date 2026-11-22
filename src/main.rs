use hopfield::net::utility::{Dataset, Utils};
use hopfield::tests::test_1_stability::StabilityTest;
use hopfield::tests::test_2_common_block::CommonBlockTest;
use hopfield::tests::test_3_recovery::RecoveryTest;
use hopfield::tests::test_4_groups::GroupsTest;
use hopfield::tests::test_5_random_input::RandomInputTest;
use hopfield::tests::test_6_own_5x5::Own5x5Test;
use hopfield::tests::test_7_large;

mod viz;

const DATA_DIR: &str = "src/data";

/// cargo run --release -- test_number_1 test_number_2 etc.
fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1).peekable();
    match args.peek().map(String::as_str) {
        Some("viz") => return viz::run(&args.skip(1).collect::<Vec<_>>()),
        Some("-h" | "--help") => {
            println!("cargo run --release -- [1 2 3 4 5 6 7]\n{}", viz::USAGE);
            return Ok(());
        }
        _ => {}
    }
    let args: Vec<usize> = args
        .map(|arg| match arg.parse() {
            Ok(n @ 1..=7) => Ok(n),
            _ => Err(format!(
                "Nieznany eksperyment {arg}. Użyj numeru 1–7 albo --help."
            )),
        })
        .collect::<Result<_, _>>()?;
    let selected = |nr: usize| args.is_empty() || args.contains(&nr);
    let datasets: Vec<Dataset> = if (1..=5).any(selected) {
        Utils::load_datasets(DATA_DIR)
    } else {
        Vec::new()
    };

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
    if selected(7) {
        test_7_large::run()?;
    }
    Ok(())
}
