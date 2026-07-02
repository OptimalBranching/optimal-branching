//! Merge-based branching-rule solvers that bypass the set-cover route.
//!
//! Faithful port of Julia's `OptimalBranchingCore/greedymerge.jl`. Where the
//! set-cover path ([`crate::set_cover::minimize_gamma`]) enumerates *candidate
//! clauses* and solves a weighted set cover, these solvers instead start from
//! the raw branching-table rows ([`bit_clauses`]) and repeatedly merge them:
//!
//! * [`NaiveBranch`] — no merging at all; one clause per table row. This is the
//!   crudest branching rule and serves as the baseline the others improve on.
//! * [`GreedyMerge`] — greedily merges pairs of rows whenever the merge lowers
//!   the overall branching factor γ, until no beneficial merge remains.
//!
//! Both implement [`BranchingRuleSolver`](crate::branch::BranchingRuleSolver)
//! directly rather than [`SetCoverSolver`](crate::solver::SetCoverSolver): they
//! produce a branching rule straight from the table without ever solving a
//! weighted set cover, so the subset/weights interface does not apply to them
//! (mirroring how Julia's `GreedyMerge`/`NaiveBranch` override
//! `optimal_branching_rule` instead of implementing the cover solve).

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

use crate::branch::{size_reduction, BranchAndReduceProblem, BranchingRuleSolver, Measure};
use crate::branching_table::{bit_mask, gather2, BranchingTable};
use crate::clause::{Clause, DNF};
use crate::error::Error;
use crate::set_cover::{complexity_bv, OptimalBranchingResult};

/// Branching-rule solver that emits one clause per branching-table row, with no
/// merging. Equivalent to Julia's `NaiveBranch`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveBranch;

/// Branching-rule solver that greedily merges table rows to shrink the
/// branching factor. Equivalent to Julia's `GreedyMerge`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreedyMerge;

/// Wrap every bitstring of every group in the branching table as a one-literal
/// clause over all `bit_length` variables.
///
/// Returns one `Vec<Clause>` per table group (row). Mirrors Julia's
/// `bit_clauses(tbl)`.
pub fn bit_clauses(table: &BranchingTable) -> Vec<Vec<Clause>> {
    let full_mask = bit_mask(table.bit_length);
    table
        .table
        .iter()
        .map(|group| group.iter().map(|&bs| Clause::new(full_mask, bs)).collect())
        .collect()
}

impl BranchingRuleSolver for NaiveBranch {
    fn optimal_branching_rule<P, M>(
        &self,
        problem: &P,
        table: &BranchingTable,
        variables: &[usize],
        measure: &M,
    ) -> Result<OptimalBranchingResult, Error>
    where
        P: BranchAndReduceProblem,
        M: Measure<P>,
    {
        // One clause per row, taken from each row's first bitstring — no merging.
        // Unlike GreedyMerge, NaiveBranch never inspects the other bitstrings, so it
        // builds the clauses directly rather than allocating the full per-row groups
        // of `bit_clauses` only to discard all but the first.
        let full_mask = bit_mask(table.bit_length);
        let clauses: Vec<Clause> = table
            .table
            .iter()
            .map(|row| Clause::new(full_mask, row[0]))
            .collect();
        let branching_vector: Vec<f64> = clauses
            .iter()
            .map(|c| size_reduction(problem, measure, c, variables).into())
            .collect();
        let gamma = complexity_bv(&branching_vector);
        Ok(OptimalBranchingResult {
            optimal_rule: DNF { clauses },
            branching_vector,
            gamma,
        })
    }
}

impl BranchingRuleSolver for GreedyMerge {
    fn optimal_branching_rule<P, M>(
        &self,
        problem: &P,
        table: &BranchingTable,
        variables: &[usize],
        measure: &M,
    ) -> Result<OptimalBranchingResult, Error>
    where
        P: BranchAndReduceProblem,
        M: Measure<P>,
    {
        let candidates = bit_clauses(table);
        Ok(greedymerge(&candidates, problem, variables, measure))
    }
}

/// Greedily merge branching-table rows to minimise the branching factor γ.
///
/// Faithful port of Julia's `greedymerge`. Each round computes γ from the
/// current per-row size reductions, then for every pair of rows evaluates the
/// best merge (the [`gather2`] of one clause from each row that maximises the
/// size reduction). Pairs whose merge lowers the energy
/// `γ^(-reduction) − w_i − w_j` are enqueued; merges are applied cheapest-first,
/// each collapsing two rows into one, until no beneficial merge remains, at
/// which point the surviving rows form the branching rule.
pub fn greedymerge<P, M>(
    cls: &[Vec<Clause>],
    problem: &P,
    variables: &[usize],
    measure: &M,
) -> OptimalBranchingResult
where
    P: BranchAndReduceProblem,
    M: Measure<P>,
{
    let nvars = variables.len();

    // Best merge of two rows: over all clause pairs, the gather2 with the
    // largest size reduction. Returns (merged_clause, reduction); an all-zero
    // clause with reduction 0 if no pair shares a non-empty mask.
    let reduction_merge = |cli: &[Clause], clj: &[Clause]| -> (Clause, f64) {
        let mut clmax = Clause::new(0, 0);
        let mut reduction_max = 0.0_f64;
        for a in cli {
            for b in clj {
                let cl12 = gather2(nvars, a, b);
                if cl12.mask == 0 {
                    continue;
                }
                let reduction: f64 = size_reduction(problem, measure, &cl12, variables).into();
                if reduction > reduction_max {
                    clmax = cl12;
                    reduction_max = reduction;
                }
            }
        }
        (clmax, reduction_max)
    };

    let mut cls: Vec<Vec<Clause>> = cls.to_vec();
    let mut size_reductions: Vec<f64> = cls
        .iter()
        .map(|group| size_reduction(problem, measure, &group[0], variables).into())
        .collect();

    loop {
        let nc = cls.len();
        let mut mask = vec![true; nc];
        let gamma = complexity_bv(&size_reductions);
        let mut weights: Vec<f64> = size_reductions.iter().map(|&s| gamma.powf(-s)).collect();

        // Priority queue over row pairs keyed by the merge energy dE (smallest,
        // i.e. most negative, first).
        let mut queue = PairQueue::new();
        for i in 0..nc {
            for j in (i + 1)..nc {
                let (_, reduction) = reduction_merge(&cls[i], &cls[j]);
                let de = gamma.powf(-reduction) - weights[i] - weights[j];
                if de <= -1e-12 {
                    queue.upsert((i, j), de);
                }
            }
        }

        if queue.is_empty() {
            let clauses: Vec<Clause> = cls.iter().map(|group| group[0]).collect();
            return OptimalBranchingResult {
                optimal_rule: DNF { clauses },
                branching_vector: size_reductions,
                gamma,
            };
        }

        while let Some((i, j)) = queue.pop_min() {
            // Drop rows i and j and every queued pair that touched them.
            for rowid in [i, j] {
                mask[rowid] = false;
                for (l, &active) in mask.iter().enumerate() {
                    if active {
                        queue.remove(ordered_pair(rowid, l));
                    }
                }
            }
            // Reinstate row i as the merged row.
            mask[i] = true;
            let (clij, reduction_i) = reduction_merge(&cls[i], &cls[j]);
            size_reductions[i] = reduction_i;
            cls[i] = vec![clij];
            weights[i] = gamma.powf(-reduction_i);
            // Re-evaluate every pair involving the new merged row.
            for (l, &active) in mask.iter().enumerate() {
                if i != l && active {
                    let (a, b) = ordered_pair(i, l);
                    let (_, reduction) = reduction_merge(&cls[a], &cls[b]);
                    let de = gamma.powf(-reduction) - weights[a] - weights[b];
                    if de <= -1e-12 {
                        queue.upsert((a, b), de);
                    }
                }
            }
        }

        // Compact away the consumed rows and try another round.
        let mut next_cls = Vec::with_capacity(nc);
        let mut next_sr = Vec::with_capacity(nc);
        for k in 0..nc {
            if mask[k] {
                next_cls.push(std::mem::take(&mut cls[k]));
                next_sr.push(size_reductions[k]);
            }
        }
        cls = next_cls;
        size_reductions = next_sr;
    }
}

#[inline]
fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Total-ordered `f64` wrapper so priorities can live in a `BTreeSet`.
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Min-priority queue of `(row, row)` pairs keyed by an `f64` priority, with
/// delete-by-key and update-in-place — the operations Julia's `PriorityQueue`
/// provides and that `greedymerge` relies on. Backed by a `BTreeSet` (ordered
/// by priority) plus a key→priority map; all operations are `O(log n)`.
struct PairQueue {
    by_priority: BTreeSet<(OrdF64, usize, usize)>,
    priority_of: HashMap<(usize, usize), OrdF64>,
}

impl PairQueue {
    fn new() -> Self {
        Self {
            by_priority: BTreeSet::new(),
            priority_of: HashMap::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.priority_of.is_empty()
    }

    /// Insert `key`, or replace its priority if already present.
    fn upsert(&mut self, key: (usize, usize), priority: f64) {
        let p = OrdF64(priority);
        if let Some(old) = self.priority_of.insert(key, p) {
            self.by_priority.remove(&(old, key.0, key.1));
        }
        self.by_priority.insert((p, key.0, key.1));
    }

    fn remove(&mut self, key: (usize, usize)) {
        if let Some(old) = self.priority_of.remove(&key) {
            self.by_priority.remove(&(old, key.0, key.1));
        }
    }

    fn pop_min(&mut self) -> Option<(usize, usize)> {
        let &(p, i, j) = self.by_priority.iter().next()?;
        self.by_priority.remove(&(p, i, j));
        self.priority_of.remove(&(i, j));
        Some((i, j))
    }
}

// ============================================================
// Tests ported from Julia OptimalBranchingCore/test/greedymerge.jl
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch::TableSolver;
    use crate::mock::{MockProblem, MockTableSolver, NumOfVariables};
    use rand::prelude::*;

    // ---- Ported from greedymerge.jl "bit_clauses" testset ----
    #[test]
    fn test_bit_clauses() {
        // Julia table: [[4, 2], [9], [20]] over 5 bits.
        let tbl = BranchingTable::new(5, vec![vec![4, 2], vec![9], vec![20]]);
        let bc = bit_clauses(&tbl);
        // Clause(bmask(1:5)=31, 4) → mask 31, val 4.
        assert_eq!(bc[0][0].mask, 31);
        assert_eq!(bc[0][0].val, 4);
        assert_eq!(bc.len(), 3);
        assert_eq!(bc[0].len(), 2);
    }

    // ---- Ported from greedymerge.jl "greedymerge large scale" testset ----
    #[test]
    fn test_greedymerge_large_scale() {
        let n = 1000; // total number of variables
        let mut rng = StdRng::seed_from_u64(2024);
        let optimal: Vec<bool> = (0..n).map(|_| rng.random()).collect();
        let p = MockProblem { optimal };

        let nvars = 18; // number of variables to branch on
        let variables: Vec<usize> = (0..nvars).collect();

        let table_solver = MockTableSolver::new(1000, 99);
        let tbl = table_solver.solve(&p, &variables);
        let candidates = bit_clauses(&tbl);

        let m = NumOfVariables;
        let result = greedymerge(&candidates, &p, &variables, &m);

        // Julia: length(tbl.table)^(1/nvars) > result.γ
        let bound = (tbl.table.len() as f64).powf(1.0 / nvars as f64);
        assert!(
            bound > result.gamma,
            "expected {bound} > gamma {}, table rows = {}",
            result.gamma,
            tbl.table.len()
        );
        assert!(result.gamma >= 1.0);
    }

    // Merging never increases the branching factor relative to no merging.
    #[test]
    fn test_greedymerge_beats_naive_gamma() {
        let mut rng = StdRng::seed_from_u64(7);
        let optimal: Vec<bool> = (0..200).map(|_| rng.random()).collect();
        let p = MockProblem { optimal };
        let variables: Vec<usize> = (0..14).collect();
        let table_solver = MockTableSolver::new(50, 123);
        let tbl = table_solver.solve(&p, &variables);

        let m = NumOfVariables;
        let naive = NaiveBranch
            .optimal_branching_rule(&p, &tbl, &variables, &m)
            .unwrap();
        let greedy = GreedyMerge
            .optimal_branching_rule(&p, &tbl, &variables, &m)
            .unwrap();

        assert!(
            greedy.gamma <= naive.gamma + 1e-9,
            "greedy gamma {} should not exceed naive gamma {}",
            greedy.gamma,
            naive.gamma
        );
    }
}
