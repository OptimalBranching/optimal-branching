use crate::error::Error;
use good_lp::*;
use rand::prelude::*;

/// Weighted minimum set cover solver.
pub trait SetCoverSolver {
    fn solve(
        &self,
        subsets: &[Vec<usize>],
        num_groups: usize,
        weights: &[f64],
    ) -> Result<Vec<usize>, Error>;

    fn max_itr(&self) -> usize;
}

/// Integer Programming solver (exact).
pub struct IPSolver {
    pub max_itr: usize,
}

impl Default for IPSolver {
    fn default() -> Self {
        Self { max_itr: 20 }
    }
}

/// LP Relaxation solver with probabilistic rounding.
pub struct LPSolver {
    pub max_itr: usize,
    pub seed: u64,
}

impl Default for LPSolver {
    fn default() -> Self {
        Self {
            max_itr: 20,
            seed: 42,
        }
    }
}

fn build_reverse_index(subsets: &[Vec<usize>], num_groups: usize) -> Vec<Vec<usize>> {
    let mut sets_id: Vec<Vec<usize>> = vec![vec![]; num_groups];
    for (i, subset) in subsets.iter().enumerate() {
        for &j in subset {
            if j < num_groups {
                sets_id[j].push(i);
            }
        }
    }
    sets_id
}

/// Solve a weighted set cover as LP or IP. Returns per-variable solution values.
fn solve_cover_model(
    subsets: &[Vec<usize>],
    num_groups: usize,
    weights: &[f64],
    integer: bool,
) -> Result<Vec<f64>, Error> {
    let nsc = subsets.len();
    if nsc == 0 {
        return if num_groups == 0 {
            Ok(vec![])
        } else {
            Err(Error::Infeasible)
        };
    }

    let sets_id = build_reverse_index(subsets, num_groups);

    let mut problem = ProblemVariables::new();
    let vars: Vec<Variable> = (0..nsc)
        .map(|_| {
            let v = variable().min(0).max(1);
            problem.add(if integer { v.integer() } else { v })
        })
        .collect();

    // The branching losses are exponentially large (often ~1e52..1e55); HiGHS's
    // MIP solver breaks down numerically on objective coefficients of that
    // magnitude and returns model status `Unknown` with no solution (which would
    // otherwise force the greedy fallback below). The weighted set-cover argmin
    // is invariant under a positive rescaling, so normalize by the max weight
    // into [.., 1] before solving.
    let wmax = weights.iter().cloned().fold(0.0_f64, f64::max);
    let scale = if wmax > 0.0 { wmax } else { 1.0 };
    let objective: Expression = vars
        .iter()
        .enumerate()
        .map(|(i, &v)| (weights[i] / scale) * v)
        .sum();
    let mut model = problem.minimise(objective).using(default_solver);

    for group_candidates in &sets_id {
        let coverage: Expression = group_candidates.iter().map(|&i| vars[i]).sum();
        model = model.with(coverage.geq(1.0));
    }

    let solution = model
        .solve()
        .map_err(|e| Error::SolverError(e.to_string()))?;

    Ok(vars.iter().map(|&v| solution.value(v)).collect())
}

impl SetCoverSolver for IPSolver {
    fn solve(
        &self,
        subsets: &[Vec<usize>],
        num_groups: usize,
        weights: &[f64],
    ) -> Result<Vec<usize>, Error> {
        match solve_cover_model(subsets, num_groups, weights, true) {
            Ok(values) => Ok((0..values.len()).filter(|&i| values[i] > 0.5).collect()),
            // Exact MIP solve failed (e.g. HiGHS returns no usable solution on a
            // degenerate cover where one subset already covers every group — its
            // presolve resolves to model status `Unknown` with no feasible
            // primal). Fall back to deterministic greedy rather than letting the
            // caller drop the branch (which would silently lose part of the MIS).
            Err(_) => Ok(greedy_set_cover(subsets, num_groups, weights)),
        }
    }

    fn max_itr(&self) -> usize {
        self.max_itr
    }
}

impl SetCoverSolver for LPSolver {
    fn solve(
        &self,
        subsets: &[Vec<usize>],
        num_groups: usize,
        weights: &[f64],
    ) -> Result<Vec<usize>, Error> {
        let fractional = solve_cover_model(subsets, num_groups, weights, false)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        Ok(pick_sets(&fractional, subsets, num_groups, &mut rng))
    }

    fn max_itr(&self) -> usize {
        self.max_itr
    }
}

/// Deterministic greedy weighted set cover.
///
/// Fallback for when the exact IP solver fails to return a solution. HiGHS can
/// terminate with a non-solution model status (`Unknown` → no feasible primal)
/// on degenerate cover MIPs — e.g. when a single subset already covers every
/// group, which its presolve resolves without ever reporting a solution. (The
/// Julia reference avoids this by short-circuiting that exact case before
/// calling the IP solver.) Greedy always produces a valid cover when one
/// exists: at each step pick the unpicked subset with the most newly-covered
/// groups per unit weight, ties broken by lowest index for determinism. The
/// cover may be non-minimal (a few extra branches downstream) but is correct.
/// If the remaining groups cannot be covered, falls back to selecting every
/// subset (safe over-cover) so the caller never silently drops a branch.
fn greedy_set_cover(subsets: &[Vec<usize>], num_groups: usize, weights: &[f64]) -> Vec<usize> {
    let nsc = subsets.len();
    let mut picked = vec![false; nsc];
    let mut covered = vec![false; num_groups];
    let mut num_covered = 0usize;

    while num_covered < num_groups {
        let mut best: Option<usize> = None;
        let mut best_ratio = f64::NEG_INFINITY;
        for i in 0..nsc {
            if picked[i] {
                continue;
            }
            let new = subsets[i]
                .iter()
                .filter(|&&g| g < num_groups && !covered[g])
                .count();
            if new == 0 {
                continue;
            }
            let ratio = new as f64 / weights[i].max(f64::MIN_POSITIVE);
            if ratio > best_ratio {
                best_ratio = ratio;
                best = Some(i);
            }
        }
        match best {
            Some(i) => {
                picked[i] = true;
                for &g in &subsets[i] {
                    if g < num_groups && !covered[g] {
                        covered[g] = true;
                        num_covered += 1;
                    }
                }
            }
            None => return (0..nsc).collect(), // remaining groups uncoverable: over-cover
        }
    }

    (0..nsc).filter(|&i| picked[i]).collect()
}

/// Probabilistic rounding of fractional LP solution.
/// Safety: bounded to 1000 * nsc rounds to avoid infinite loop on degenerate inputs.
fn pick_sets(
    fractional: &[f64],
    subsets: &[Vec<usize>],
    num_groups: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let nsc = fractional.len();
    let mut picked = vec![false; nsc];
    let mut num_covered = 0usize;
    let mut covered = vec![false; num_groups];
    let max_rounds = 1000 * nsc.max(1);

    for _ in 0..max_rounds {
        for i in 0..nsc {
            if !picked[i] && rng.random::<f64>() < fractional[i] {
                picked[i] = true;
                for &g in &subsets[i] {
                    if !covered[g] {
                        covered[g] = true;
                        num_covered += 1;
                    }
                }
                if num_covered == num_groups {
                    return picked
                        .iter()
                        .enumerate()
                        .filter(|(_, &p)| p)
                        .map(|(i, _)| i)
                        .collect();
                }
            }
        }
    }
    (0..nsc).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ip_solver_simple() {
        let subsets = vec![vec![0], vec![1], vec![0, 1]];
        let weights = vec![1.0, 1.0, 1.5];
        let solver = IPSolver { max_itr: 20 };
        let selected = solver.solve(&subsets, 2, &weights).unwrap();
        assert_eq!(selected, vec![2]);
    }

    #[test]
    fn test_ip_solver_must_cover_all() {
        let subsets = vec![vec![0], vec![1]];
        let weights = vec![1.0, 1.0];
        let solver = IPSolver { max_itr: 20 };
        let selected = solver.solve(&subsets, 2, &weights).unwrap();
        let mut sorted = selected.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1]);
    }

    #[test]
    fn test_lp_solver_covers_all() {
        let subsets = vec![vec![0], vec![1], vec![0, 1]];
        let weights = vec![1.0, 1.0, 1.5];
        let solver = LPSolver {
            max_itr: 20,
            seed: 42,
        };
        let selected = solver.solve(&subsets, 2, &weights).unwrap();
        let mut covered = std::collections::HashSet::new();
        for &i in &selected {
            for &g in &subsets[i] {
                covered.insert(g);
            }
        }
        assert_eq!(covered.len(), 2);
    }

    fn covers_all(selected: &[usize], subsets: &[Vec<usize>], num_groups: usize) -> bool {
        let mut covered = std::collections::HashSet::new();
        for &i in selected {
            for &g in &subsets[i] {
                covered.insert(g);
            }
        }
        (0..num_groups).all(|g| covered.contains(&g))
    }

    #[test]
    fn test_greedy_covers_all_and_is_deterministic() {
        let subsets = vec![vec![0], vec![1], vec![0, 1], vec![2], vec![1, 2]];
        let weights = vec![1.0, 1.0, 1.5, 1.0, 1.2];
        let a = greedy_set_cover(&subsets, 3, &weights);
        let b = greedy_set_cover(&subsets, 3, &weights);
        assert_eq!(a, b, "greedy must be deterministic");
        assert!(covers_all(&a, &subsets, 3));
    }

    #[test]
    fn test_greedy_prefers_full_cover_set() {
        // The degenerate case that breaks HiGHS: one subset covers everything.
        let subsets = vec![vec![0], vec![1], vec![2], vec![0, 1, 2]];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let selected = greedy_set_cover(&subsets, 3, &weights);
        assert_eq!(selected, vec![3]);
    }
}
