//! Generic branch-and-reduce framework that turns the problem of finding
//! the *optimal* branching rule into an instance of weighted set cover.
//!
//! The original method is described in *Automated discovery of optimal
//! branching rules for the branch-and-bound algorithm* (arXiv:
//! [2412.07685](https://arxiv.org/abs/2412.07685)).
//!
//! # Quick start
//!
//! ```no_run
//! use optimal_branching::{
//!     branch_and_reduce, BranchingStrategy, MaxSize, NoReducer,
//!     mock::{MockProblem, MockTableSolver, NumOfVariables, RandomSelector},
//!     solver::IPSolver,
//! };
//!
//! let problem = MockProblem { optimal: vec![true, false, true] };
//! let strategy = BranchingStrategy::new(
//!     MockTableSolver { n: 16, p: 0.3, seed: 42 },
//!     IPSolver::default(),
//!     RandomSelector { n: 2, seed: 42 },
//!     NumOfVariables,
//!     NoReducer,
//! );
//! let answer: MaxSize = branch_and_reduce(problem, &strategy).unwrap();
//! assert!(answer.size <= 2.0);
//! ```
//!
//! # Implementing your own problem
//!
//! See the [`mock`] module for a minimal worked example, and
//! `examples/mock_problem.rs` for the same example as a runnable
//! binary.

pub mod branch;
pub mod branching_table;
pub mod clause;
pub mod error;
pub mod greedymerge;
pub mod mock;
pub mod set_cover;
pub mod solver;

pub use branch::{
    branch_and_reduce, optimal_branching_rule, size_reduction, BranchAndReduceProblem,
    BranchingRuleSolver, BranchingStrategy, MaxSize, MaxSizeBranchCount, Measure, NoReducer,
    Reducer, ReductionResult, ResultAlgebra, Selector, TableSolver, VertexDecoding,
};
pub use branching_table::{BranchingTable, CandidateClause};
pub use clause::{Clause, DNF};
pub use error::Error;
pub use greedymerge::{bit_clauses, greedymerge, GreedyMerge, NaiveBranch};
pub use set_cover::{complexity_bv, minimize_gamma, OptimalBranchingResult};
pub use solver::{IPSolver, LPSolver, SetCoverSolver};
