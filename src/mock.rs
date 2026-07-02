//! A minimal, self-contained example problem.
//!
//! The types in this module are a faithful port of Julia's
//! `OptimalBranchingCore/mockproblem.jl`. They are not a benchmark — they
//! exist so downstream users can see how to implement
//! [`crate::BranchAndReduceProblem`], [`crate::Measure`],
//! [`crate::Selector`], [`crate::TableSolver`], and [`crate::Reducer`]
//! for their own problems with one focused file as reference.
//!
//! # Examples
//!
//! See `examples/mock_problem.rs` for a runnable walkthrough.

use rand::prelude::*;
use std::collections::HashSet;

pub use crate::branch::NoReducer;

use crate::branch::{BranchAndReduceProblem, Measure, Selector, TableSolver};
use crate::branching_table::BranchingTable;
use crate::clause::Clause;

/// A toy problem whose "optimal solution" is a fixed boolean vector.
///
/// Branching removes variables one at a time; the local value of a branch is
/// the number of correctly-assigned variables it fixes.
#[derive(Debug, Clone)]
pub struct MockProblem {
    pub optimal: Vec<bool>,
}

/// Measure that counts the number of remaining variables.
///
/// Julia: `NumOfVariables`
pub struct NumOfVariables;

impl Measure<MockProblem> for NumOfVariables {
    type Output = f64;
    fn measure(&self, p: &MockProblem) -> f64 {
        p.optimal.len() as f64
    }
    fn delta(&self, _p: &MockProblem, removed: &[usize]) -> f64 {
        removed.len() as f64
    }
}

/// Selector that picks `n` variables at random using a fixed seed.
///
/// Julia: `RandomSelector`
pub struct RandomSelector {
    pub n: usize,
    pub seed: u64,
}

impl Selector<MockProblem> for RandomSelector {
    fn select(&self, p: &MockProblem, _measure: &impl Measure<MockProblem>) -> Vec<usize> {
        let nv = self.n.min(p.optimal.len());
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(p.optimal.len() as u64));
        let mut indices: Vec<usize> = (0..p.optimal.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(nv);
        indices.sort();
        indices
    }
}

/// Table solver that generates random branching tables.
///
/// Julia: `MockTableSolver`
pub struct MockTableSolver {
    pub n: usize,
    pub p: f64,
    pub seed: u64,
}

impl MockTableSolver {
    pub fn new(n: usize, seed: u64) -> Self {
        Self { n, p: 0.0, seed }
    }
}

impl TableSolver<MockProblem> for MockTableSolver {
    fn solve(&self, problem: &MockProblem, variables: &[usize]) -> BranchingTable {
        let nvars = variables.len();
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(problem.optimal.len() as u64));

        // rand_fib: random independent set on 1D chain
        let rand_fib = |rng: &mut StdRng| -> Vec<bool> {
            let mut bs = vec![false; nvars];
            for i in 0..nvars {
                let threshold = if i == 0 {
                    0.5_f64.min(1.0)
                } else {
                    0.5_f64.min(if bs[i - 1] { 0.0 } else { 1.0 })
                };
                if rng.random::<f64>() < threshold {
                    bs[i] = true;
                }
            }
            bs
        };

        let bool_to_u64 = |bs: &[bool]| -> u64 {
            let mut v = 0u64;
            for (i, &b) in bs.iter().enumerate() {
                if b {
                    v |= 1u64 << i;
                }
            }
            v
        };

        // Generate rows, ensuring optimal is included
        let mut rows_set: HashSet<u64> = HashSet::new();
        let mut rows: Vec<Vec<u64>> = Vec::new();
        for _ in 0..self.n {
            let bs = rand_fib(&mut rng);
            let v = bool_to_u64(&bs);
            if rows_set.insert(v) {
                rows.push(vec![v]);
            }
        }
        // Add optimal solution
        let opt_bs: Vec<bool> = variables.iter().map(|&i| problem.optimal[i]).collect();
        let opt_v = bool_to_u64(&opt_bs);
        if rows_set.insert(opt_v) {
            rows.push(vec![opt_v]);
        }

        // Add extra bitstrings per row with probability p
        for row in &mut rows {
            for _ in 0..100 {
                if rng.random::<f64>() < self.p {
                    row.push(bool_to_u64(&rand_fib(&mut rng)));
                } else {
                    break;
                }
            }
            row.sort();
            row.dedup();
        }

        BranchingTable::new(nvars, rows)
    }
}

impl BranchAndReduceProblem for MockProblem {
    type LocalValue = f64;

    fn is_empty(&self) -> bool {
        self.optimal.is_empty()
    }

    /// Julia: `apply_branch(p::MockProblem, clause, variables)`
    fn apply_branch(&self, clause: &Clause, variables: &[usize]) -> (Self, f64) {
        let mut remain_mask = vec![true; self.optimal.len()];
        let mut correct_count = 0.0;
        for (i, &var_idx) in variables.iter().enumerate() {
            if ((clause.mask >> i) & 1) == 1 {
                remain_mask[var_idx] = false;
                let val_bit = (clause.val >> i) & 1;
                if (val_bit == 1) == self.optimal[var_idx] {
                    correct_count += 1.0;
                }
            }
        }
        let new_optimal: Vec<bool> = self
            .optimal
            .iter()
            .enumerate()
            .filter(|(i, _)| remain_mask[*i])
            .map(|(_, &v)| v)
            .collect();
        (
            MockProblem {
                optimal: new_optimal,
            },
            correct_count,
        )
    }
}
