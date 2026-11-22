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
fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("viz") => viz::run(&args[1..]),
        Some("-h" | "--help") => {
            println!(
                "cargo run --release -- [1 2 3 4 5 6 7]\ncargo run --release -- {}",
                viz::USAGE
            );
            Ok(())
        }
        _ => experiments(&args),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn experiments(args: &[String]) -> Result<(), String> {
    let args: Vec<usize> = args
        .iter()
        .map(|arg| {
            arg.parse::<usize>()
                .ok()
                .filter(|n| (1..=7).contains(n))
                .ok_or_else(|| format!("Nieznany eksperyment {arg}. Użyj numeru 1–7 albo --help."))
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
