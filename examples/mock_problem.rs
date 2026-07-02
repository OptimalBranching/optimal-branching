//! Runnable end-to-end demo of the optimal-branching API on the
//! synthetic [`mock::MockProblem`].
//!
//! Run: `cargo run --release -p optimal-branching --example mock_problem`
//!
//! Prints a one-line summary of the form
//!   `MockProblem optimal size = …, branches taken = …`
//! Exact numbers depend on the random table-solver's seed.

use optimal_branching::mock::{
    MockProblem, MockTableSolver, NoReducer, NumOfVariables, RandomSelector,
};
use optimal_branching::solver::IPSolver;
use optimal_branching::{branch_and_reduce, BranchingStrategy, MaxSizeBranchCount};

fn main() {
    let problem = MockProblem {
        optimal: vec![true, false, true, true, false, true, true],
    };

    let strategy = BranchingStrategy::new(
        MockTableSolver {
            n: 32,
            p: 0.3,
            seed: 42,
        },
        IPSolver::default(),
        RandomSelector { n: 4, seed: 42 },
        NumOfVariables,
        NoReducer,
    );

    let result: MaxSizeBranchCount =
        branch_and_reduce(problem, &strategy).expect("branch_and_reduce returned an error");

    println!(
        "MockProblem optimal size = {}, branches taken = {}",
        result.size as usize, result.count
    );
}
