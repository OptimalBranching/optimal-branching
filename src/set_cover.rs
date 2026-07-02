use std::collections::HashMap;

use crate::branching_table::{BranchingTable, CandidateClause};
use crate::clause::{Clause, DNF};
use crate::error::Error;
use crate::solver::SetCoverSolver;

/// Result of optimal branching rule search.
#[derive(Debug, Clone)]
pub struct OptimalBranchingResult {
    pub optimal_rule: DNF,
    pub branching_vector: Vec<f64>,
    pub gamma: f64,
}

/// Compute branching factor gamma from a branching vector.
/// Solves: sum(gamma^(-delta_rho_i)) = 1 via bisection.
pub fn complexity_bv(branching_vector: &[f64]) -> f64 {
    if branching_vector.is_empty() {
        return 0.0;
    }
    if branching_vector.iter().any(|&x| x <= 0.0) {
        return f64::INFINITY;
    }

    let f = |x: f64| -> f64 { branching_vector.iter().map(|&d| x.powf(-d)).sum::<f64>() - 1.0 };

    let a = 1.0_f64;
    let fa = f(a);

    let mut b = 2.0_f64;
    for _ in 0..10 {
        if f(b) < 0.0 {
            break;
        }
        b *= 2.0;
    }
    if f(b) > 0.0 {
        return f64::INFINITY;
    }

    bisect_solve(f, a, fa, b, f(b))
}

pub(crate) fn bisect_solve(
    f: impl Fn(f64) -> f64,
    mut a: f64,
    mut fa: f64,
    mut b: f64,
    fb: f64,
) -> f64 {
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }
    while b - a > f64::EPSILON * b {
        let c = (a + b) / 2.0;
        let fc = f(c);
        if fc == 0.0 {
            return c;
        } else if fa * fc < 0.0 {
            b = c;
        } else {
            a = c;
            fa = fc;
        }
    }
    (a + b) / 2.0
}

/// Remove strictly dominated candidates.
/// If two candidates cover identical groups, keep the one with higher delta_rho.
pub fn remove_dominated(subsets: &[Vec<usize>], delta_rho: &[f64]) -> Vec<usize> {
    let mut dict: HashMap<&[usize], (usize, f64)> = HashMap::new();
    let mut mask = vec![true; subsets.len()];

    for i in 0..subsets.len() {
        let key: &[usize] = &subsets[i];
        if let Some(&(prev_idx, prev_dr)) = dict.get(key) {
            if delta_rho[i] <= prev_dr {
                mask[i] = false;
            } else {
                mask[prev_idx] = false;
                dict.insert(key, (i, delta_rho[i]));
            }
        } else {
            dict.insert(key, (i, delta_rho[i]));
        }
    }

    mask.iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .map(|(i, _)| i)
        .collect()
}

/// Iteratively find the branching rule that minimizes gamma.
pub fn minimize_gamma(
    table: &BranchingTable,
    candidates: &[CandidateClause],
    delta_rho: &[f64],
    solver: &impl SetCoverSolver,
) -> Result<OptimalBranchingResult, Error> {
    let num_groups = table.num_groups();

    let subsets: Vec<Vec<usize>> = candidates.iter().map(|c| c.covered_items.clone()).collect();

    // Eliminate dominated candidates
    let kept = remove_dominated(&subsets, delta_rho);
    let kept_delta_rho: Vec<f64> = kept.iter().map(|&i| delta_rho[i]).collect();

    // Short-circuit: if any single candidate covers all groups
    for (k, &ki) in kept.iter().enumerate() {
        if subsets[ki].len() == num_groups {
            return Ok(OptimalBranchingResult {
                optimal_rule: DNF {
                    clauses: vec![candidates[ki].clause],
                },
                branching_vector: vec![kept_delta_rho[k]],
                gamma: 1.0,
            });
        }
    }

    let kept_subsets: Vec<Vec<usize>> = kept.iter().map(|&i| subsets[i].clone()).collect();

    let max_itr = solver.max_itr();
    let mut gamma_old = 2.0_f64;
    let mut picked_scs = Vec::new();
    let mut best_gamma = f64::INFINITY;
    let mut best_picked_scs: Vec<usize> = Vec::new();

    for iter in 0..max_itr {
        let weights: Vec<f64> = kept_delta_rho
            .iter()
            .map(|&dr| 1.0 / gamma_old.powf(dr))
            .collect();

        picked_scs = solver.solve(&kept_subsets, num_groups, &weights)?;

        let picked_bv: Vec<f64> = picked_scs.iter().map(|&i| kept_delta_rho[i]).collect();
        let gamma_new = complexity_bv(&picked_bv);

        // Track best solution found so far
        if gamma_new < best_gamma {
            best_gamma = gamma_new;
            best_picked_scs = picked_scs.clone();
        }

        if (gamma_new - gamma_old).abs() < 1e-6 * gamma_old.abs().max(1.0) {
            gamma_old = gamma_new;
            break;
        }
        gamma_old = gamma_new;

        if iter == max_itr - 1 {
            // Return best solution found rather than failing
            picked_scs = best_picked_scs.clone();
            gamma_old = best_gamma;
        }
    }

    let result_clauses: Vec<Clause> = picked_scs
        .iter()
        .map(|&i| candidates[kept[i]].clause)
        .collect();
    let result_bv: Vec<f64> = picked_scs.iter().map(|&i| kept_delta_rho[i]).collect();

    Ok(OptimalBranchingResult {
        optimal_rule: DNF {
            clauses: result_clauses,
        },
        branching_vector: result_bv,
        gamma: gamma_old,
    })
}

// ============================================================
// Tests ported 1:1 from Julia OptimalBranchingCore/test/setcovering.jl
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::branching_table::BranchingTable;
    use crate::solver::{IPSolver, LPSolver};

    // ---- Ported from setcovering.jl "bisect_solve" testset ----
    #[test]
    fn test_bisect_solve() {
        let f = |x: f64| x * x - 2.0;
        let result = bisect_solve(f, 1.0, f(1.0), 2.0, f(2.0));
        assert!((result - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    /// Shared helper: run minimize_gamma with both IP and LP solvers, assert
    /// branching vectors match and both gammas are close to `expected_gamma`.
    fn assert_ip_lp_agree(tbl: &BranchingTable, expected_gamma: f64) {
        let clauses_raw = tbl.candidate_clauses();
        let delta_rho: Vec<f64> = clauses_raw.iter().map(|c| c.clause.len() as f64).collect();

        let result_ip =
            minimize_gamma(tbl, &clauses_raw, &delta_rho, &IPSolver { max_itr: 10 }).unwrap();
        let result_lp = minimize_gamma(
            tbl,
            &clauses_raw,
            &delta_rho,
            &LPSolver {
                max_itr: 10,
                seed: 42,
            },
        )
        .unwrap();

        let mut bv_ip = result_ip.branching_vector.clone();
        let mut bv_lp = result_lp.branching_vector.clone();
        bv_ip.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bv_lp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(bv_ip.len(), bv_lp.len());
        for (a, b) in bv_ip.iter().zip(bv_lp.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!((result_ip.gamma - expected_gamma).abs() < 1e-6);
        assert!((result_lp.gamma - expected_gamma).abs() < 1e-6);
    }

    #[test]
    fn test_setcover_static_bitvector_gamma_1() {
        let tbl = BranchingTable::new(5, vec![vec![4, 2], vec![9], vec![20]]);
        assert_ip_lp_agree(&tbl, 1.0);
    }

    #[test]
    fn test_setcover_static_bitvector_gamma_nontrivial() {
        let tbl = BranchingTable::new(5, vec![vec![10, 6], vec![11], vec![20]]);
        assert_ip_lp_agree(&tbl, 1.1673039782614185);
    }

    #[test]
    fn test_setcover_normal_vector() {
        let tbl = BranchingTable::new(5, vec![vec![10, 6], vec![11], vec![20]]);
        let clauses_raw = tbl.candidate_clauses();
        let delta_rho: Vec<f64> = clauses_raw.iter().map(|c| c.clause.len() as f64).collect();

        let result_ip =
            minimize_gamma(&tbl, &clauses_raw, &delta_rho, &IPSolver { max_itr: 10 }).unwrap();
        let result_lp = minimize_gamma(
            &tbl,
            &clauses_raw,
            &delta_rho,
            &LPSolver {
                max_itr: 10,
                seed: 42,
            },
        )
        .unwrap();

        // This test additionally checks covered_by (not checked in the shared helper)
        assert!(tbl.covered_by(&result_ip.optimal_rule));
        assert!(tbl.covered_by(&result_lp.optimal_rule));
        assert_ip_lp_agree(&tbl, 1.1673039782614185);
    }

    // ---- Ported from setcovering.jl "corner case (exist a clause that covers all items)" ----
    // Julia table: [[10, 6], [11], [28]]
    // [0,0,1,1,1] → bit2=1,bit3=1,bit4=1 = 4+8+16 = 28
    #[test]
    fn test_setcover_corner_case_all_covered() {
        let tbl = BranchingTable::new(5, vec![vec![10, 6], vec![11], vec![28]]);
        let clauses_raw = tbl.candidate_clauses();
        let clauses: Vec<Clause> = clauses_raw.iter().map(|c| c.clause).collect();
        let delta_rho: Vec<f64> = clauses.iter().map(|c| c.len() as f64).collect();

        let result_ip =
            minimize_gamma(&tbl, &clauses_raw, &delta_rho, &IPSolver { max_itr: 10 }).unwrap();
        assert!(tbl.covered_by(&result_ip.optimal_rule));
        assert!((result_ip.gamma - 1.0).abs() < 1e-6);
    }

    // NOTE: "intersection of clauses" and "folding clauses" tests from Julia
    // are skipped — intersect_clauses and folding_clauses are not yet implemented.

    #[test]
    fn test_complexity_bv_single() {
        assert!((complexity_bv(&[1.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_complexity_bv_two_equal() {
        assert!((complexity_bv(&[1.0, 1.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_complexity_bv_negative_returns_inf() {
        assert!(complexity_bv(&[-1.0, 1.0]).is_infinite());
    }

    #[test]
    fn test_complexity_bv_zero_returns_inf() {
        assert!(complexity_bv(&[0.0, 1.0]).is_infinite());
    }

    #[test]
    fn test_complexity_bv_empty() {
        assert!((complexity_bv(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_complexity_bv_known_value() {
        let gamma = complexity_bv(&[1.0, 2.0]);
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((gamma - golden).abs() < 1e-6);
    }

    #[test]
    fn test_remove_dominated() {
        let subsets = vec![vec![0, 1], vec![0], vec![0, 1]];
        let delta_rho = vec![1.0, 2.0, 3.0];
        let kept = remove_dominated(&subsets, &delta_rho);
        assert!(kept.contains(&1));
        assert!(kept.contains(&2));
        assert!(!kept.contains(&0));
    }

    #[test]
    fn test_minimize_gamma_single_covering() {
        let table = BranchingTable::new(2, vec![vec![0b01], vec![0b10]]);
        let candidates = vec![crate::branching_table::CandidateClause {
            clause: crate::clause::Clause::new(0, 0),
            covered_items: vec![0, 1],
        }];
        let delta_rho = vec![2.0];
        let solver = IPSolver::default();
        let result = minimize_gamma(&table, &candidates, &delta_rho, &solver).unwrap();
        assert!((result.gamma - 1.0).abs() < 1e-10);
    }
}
