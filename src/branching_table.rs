use std::collections::{HashMap, HashSet};

use crate::clause::{Clause, DNF};

#[derive(Debug, Clone)]
pub struct BranchingTable {
    pub bit_length: usize,    // num of branching variables
    pub table: Vec<Vec<u64>>, // groups of bitstrings, eg. [`0010`, `1001`, `1100`]
}

impl BranchingTable {
    pub fn new(bit_length: usize, table: Vec<Vec<u64>>) -> Self {
        Self { bit_length, table }
    }

    pub fn num_groups(&self) -> usize {
        self.table.len()
    }

    pub fn covered_items(&self, clause: &Clause) -> Vec<usize> {
        self.table
            .iter()
            .enumerate()
            .filter(|(_, group)| group.iter().any(|bs| clause.covered_by(*bs)))
            .map(|(i, _)| i)
            .collect()
    }

    /// Check which groups in the table are covered by the DNF.
    /// Here `cover` means a more general constraint (has less 1s)
    pub fn covered_by(&self, dnf: &DNF) -> bool {
        self.table
            .iter()
            .all(|group| group.iter().any(|bs| dnf.covered_by(*bs)))
    }

    /// Worklist-based candidate clause generation.
    /// Mirrors Julia's `candidate_clauses()` in setcovering.jl:305-327.
    pub fn candidate_clauses(&self) -> Vec<CandidateClause> {
        let full_mask = bit_mask(self.bit_length);

        let mut all_clauses: HashMap<Clause, Vec<usize>> = HashMap::new();
        let mut stack: Vec<Clause> = Vec::new();

        for group in &self.table {
            for &bs in group {
                stack.push(Clause::new(full_mask, bs));
            }
        }

        while let Some(c) = stack.pop() {
            if all_clauses.contains_key(&c) {
                continue;
            }

            let covered = self.covered_items(&c);
            // Build a set of covered group indices for O(1) lookup
            let covered_set: HashSet<usize> = covered.iter().copied().collect();
            all_clauses.insert(c, covered);

            for (gi, group) in self.table.iter().enumerate() {
                if !covered_set.contains(&gi) {
                    for &bs in group {
                        let c_new = gather2_with_mask(full_mask, &c, &Clause::new(full_mask, bs));
                        if c_new != c && c_new.mask != 0 && !all_clauses.contains_key(&c_new) {
                            stack.push(c_new);
                        }
                    }
                }
            }
        }

        all_clauses
            .into_iter()
            .map(|(clause, covered_items)| CandidateClause {
                clause,
                covered_items,
            })
            .collect()
    }

    /// Find a single clause consistent with all rows of the branching table.
    /// This is a "free" reduction that covers the whole table without branching.
    /// Mirrors Julia's `intersect_clauses_dfs` in setcovering.jl:196-278.
    pub fn intersect_clauses_dfs(&self) -> Option<Clause> {
        let n = self.bit_length;
        let bss = &self.table;
        if bss.is_empty() {
            return None;
        }
        let mask = bit_mask(n);
        let tbl_clauses: Vec<Vec<Clause>> = bss
            .iter()
            .map(|bs| bs.iter().map(|&b| Clause::new(mask, b)).collect())
            .collect();
        if tbl_clauses.len() == 1 {
            return tbl_clauses[0].first().copied();
        }
        let mut sorted_indices: Vec<usize> = (0..tbl_clauses.len()).collect();
        sorted_indices.sort_by_key(|&i| tbl_clauses[i].len());
        let sorted_refs: Vec<&Vec<Clause>> =
            sorted_indices.iter().map(|&i| &tbl_clauses[i]).collect();
        for c0 in sorted_refs[0] {
            if let Some(result) = intersect_dfs(&sorted_refs[1..], *c0, n) {
                return Some(result);
            }
        }
        None
    }
}

fn intersect_dfs(rows: &[&Vec<Clause>], c0: Clause, n: usize) -> Option<Clause> {
    let is_last = rows.len() == 1;
    for ci in rows[0] {
        let c_new = gather2(n, &c0, ci);
        if c_new.mask != 0 {
            if is_last {
                return Some(c_new);
            }
            if let Some(result) = intersect_dfs(&rows[1..], c_new, n) {
                return Some(result);
            }
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct CandidateClause {
    pub clause: Clause,
    pub covered_items: Vec<usize>,
}

/// Compute a bitmask with the lowest `n` bits set.
pub(crate) fn bit_mask(n: usize) -> u64 {
    assert!(n <= 64, "bit_mask(n={n}) exceeds 64-variable limit");
    if n == 0 {
        0
    } else if n == 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

/// Merge two clauses: keep bits where they agree, drop where they differ.
/// Restricts to bit_length bits to avoid stray high bits from NOT.
pub fn gather2(bit_length: usize, c1: &Clause, c2: &Clause) -> Clause {
    gather2_with_mask(bit_mask(bit_length), c1, c2)
}

fn gather2_with_mask(len_mask: u64, c1: &Clause, c2: &Clause) -> Clause {
    let agree = !(c1.val ^ c2.val);
    let new_mask = c1.mask & c2.mask & agree & len_mask;
    Clause::new(new_mask, c1.val & new_mask)
}

// ============================================================
// Tests ported 1:1 from Julia OptimalBranchingCore/test/branching_table.jl + branch.jl
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_branching_table_basic_operations() {
        let tbl = BranchingTable::new(5, vec![vec![4, 2], vec![9], vec![20]]);
        assert_eq!(tbl.bit_length, 5);
        assert_eq!(tbl.table, vec![vec![4, 2], vec![9], vec![20]]);
        assert_eq!(tbl.num_groups(), 3);
    }

    #[test]
    fn test_covered_items_step_by_step() {
        // 5-bit bitstrings:
        //   4  = 00100
        //   2  = 00010
        //   9  = 01001
        //   20 = 10100
        let tbl = BranchingTable::new(5, vec![vec![4, 2], vec![9], vec![20]]);
        // group 0: [4, 2],  group 1: [9],  group 2: [20]

        // Clause: mask=0b00100, val=0b00100 → "bit2 must be 1"
        let c1 = Clause::new(0b00100, 0b00100);
        //   group 0: 4=00100 → bit2=1 ✓ (match!)
        //   group 1: 9=01001 → bit2=0 ✗
        //   group 2: 20=10100 → bit2=1 ✓ (match!)
        assert_eq!(tbl.covered_items(&c1), vec![0, 2]);

        // Clause: mask=0b00010, val=0b00010 → "bit1 must be 1"
        let c2 = Clause::new(0b00010, 0b00010);
        //   group 0: 4=00100 → bit1=0 ✗, 2=00010 → bit1=1 ✓
        //   group 1: 9=01001 → bit1=0 ✗
        //   group 2: 20=10100 → bit1=0 ✗
        assert_eq!(tbl.covered_items(&c2), vec![0]);

        // Clause: mask=0b01001, val=0b01001 → "bit0=1 AND bit3=1"
        let c3 = Clause::new(0b01001, 0b01001);
        //   group 0: 4=00100 → bit0=0 ✗, 2=00010 → bit0=0 ✗
        //   group 1: 9=01001 → bit0=1,bit3=1 ✓
        //   group 2: 20=10100 → bit0=0 ✗
        assert_eq!(tbl.covered_items(&c3), vec![1]);

        // Clause: mask=0b00000, val=0b00000 → "no constraint" (matches nothing, mask=0)
        let c4 = Clause::new(0b00000, 0b00000);
        // mask=0 → (any_bs & 0) == 0 always true → all groups covered
        assert_eq!(tbl.covered_items(&c4), vec![0, 1, 2]);
    }

    // Julia: _vec2int maps [v1,v2,...,vn] → v1*1 + v2*2 + v3*4 + ...
    // [0,0,1,0,0] → 4, [0,1,0,0,0] → 2, [1,0,0,1,0] → 9, [0,0,1,0,1] → 20

    // ---- Ported from branching_table.jl ----

    #[test]
    fn test_branching_table_equality() {
        // Julia: tbl_1 and tbl_2 have same groups with different order within groups
        // tbl_1: group0 = [4, 2], tbl_2: group0 = [2, 4]
        // Julia BranchingTable equality uses Set comparison per group
        let tbl_1 = BranchingTable::new(5, vec![vec![4, 2], vec![9], vec![20]]);
        let tbl_2 = BranchingTable::new(5, vec![vec![2, 4], vec![9], vec![20]]);
        // Compare as sets per group
        for (g1, g2) in tbl_1.table.iter().zip(tbl_2.table.iter()) {
            let s1: HashSet<_> = g1.iter().collect();
            let s2: HashSet<_> = g2.iter().collect();
            assert_eq!(s1, s2);
        }
    }

    // ---- Ported from branch.jl "constructing candidate_clauses" ----

    /// Naive candidate clause generation for comparison.
    /// Julia: all_clauses_naive + subcovers_naive
    fn all_clauses_naive(n: usize, bss: &[Vec<u64>]) -> Vec<Clause> {
        let mut all_clauses = Vec::new();

        // Generate all combinations: pick 0 or 1 bitstring from each group
        // indices[i] ranges from 0 (skip group) to group.len() (pick group[idx-1])
        let group_sizes: Vec<usize> = bss.iter().map(|g| g.len() + 1).collect();
        let mut indices = vec![0usize; bss.len()];
        loop {
            // Collect picked bitstrings
            let cbs: Vec<u64> = bss
                .iter()
                .zip(indices.iter())
                .filter(|(_, &idx)| idx > 0)
                .map(|(group, &idx)| group[idx - 1])
                .collect();

            if !cbs.is_empty() {
                let ccbs = cover_clause(n, &cbs);
                if ccbs.mask != 0 && !all_clauses.contains(&ccbs) {
                    all_clauses.push(ccbs);
                }
            }

            // Increment indices (odometer-style)
            let mut carry = true;
            for i in 0..indices.len() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= group_sizes[i] {
                        indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }
        all_clauses
    }

    /// Julia: cover_clause — return a clause covering all given bitstrings
    fn cover_clause(n: usize, bitstrings: &[u64]) -> Clause {
        let full_mask = (1u64 << n) - 1;
        let mut mask = full_mask;
        for i in 0..bitstrings.len() - 1 {
            mask &= bitstrings[i] ^ flip_all(n, bitstrings[i + 1]);
        }
        let val = bitstrings[0] & mask;
        Clause::new(mask, val)
    }

    fn flip_all(n: usize, b: u64) -> u64 {
        b ^ ((1u64 << n) - 1)
    }

    fn subcovers_naive(tbl: &BranchingTable) -> Vec<Vec<usize>> {
        let all_clauses = all_clauses_naive(tbl.bit_length, &tbl.table);
        all_clauses.iter().map(|c| tbl.covered_items(c)).collect()
    }

    #[test]
    fn test_candidate_clauses_vs_naive() {
        // Julia table: [[4, 2], [9], [20]]
        let tbl = BranchingTable::new(5, vec![vec![4, 2], vec![9], vec![20]]);

        // Julia: is_valid, gamma = test_rule(tbl, DNF([Clause(2, 1)]), MockProblem(rand(Bool, 5)), NumOfVariables(), collect(1:5))
        // Clause(2, 1) → mask=2, val=1&2=0 → "bit1=0"
        // test_rule checks: is_valid = covered_by(tbl, dnf), gamma = complexity_bv(size_reductions)
        let rule = DNF {
            clauses: vec![Clause::new(2, 1)],
        };
        // @test is_valid
        assert!(tbl.covered_by(&rule));
        // @test gamma == 1.0
        // The clause has mask=2 (1 bit set), so size_reduction = 1.0, complexity_bv([1.0]) = 1.0
        {
            use crate::set_cover::complexity_bv;
            let size_reductions: Vec<f64> = rule.clauses.iter().map(|c| c.len() as f64).collect();
            let gamma = complexity_bv(&size_reductions);
            assert!((gamma - 1.0).abs() < 1e-10);
        }

        let clauses = tbl.candidate_clauses();
        let subsets: Vec<Vec<usize>> = clauses
            .iter()
            .map(|c| tbl.covered_items(&c.clause))
            .collect();
        let subsets_naive = subcovers_naive(&tbl);

        assert_eq!(subsets.len(), subsets_naive.len());
        let subsets_set: HashSet<Vec<usize>> = subsets.into_iter().collect();
        for sc in &subsets_naive {
            assert!(subsets_set.contains(sc));
        }
    }

    // ---- Ported from branch.jl "complexity" testset ----
    #[test]
    fn test_complexity_bv_random() {
        use crate::set_cover::complexity_bv;
        use rand::prelude::*;

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let bv: Vec<f64> = (0..5).map(|_| rng.random_range(1..=10) as f64).collect();
            let gamma = complexity_bv(&bv);
            if gamma <= 2.0 {
                // Verify: sum(gamma^(-d) for d in bv) ≈ 1.0
                let sum: f64 = bv.iter().map(|&d| gamma.powf(-d)).sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "complexity_bv({bv:?}) = {gamma}, sum = {sum}"
                );
            }
        }
    }

    #[test]
    fn test_intersect_clauses_dfs_found() {
        let tbl = BranchingTable::new(3, vec![vec![0b101, 0b010], vec![0b101, 0b011]]);
        let result = tbl.intersect_clauses_dfs();
        assert!(result.is_some());
        assert!(result.unwrap().mask != 0);
    }

    #[test]
    fn test_intersect_clauses_dfs_single_row() {
        let tbl = BranchingTable::new(2, vec![vec![0b01, 0b10]]);
        let result = tbl.intersect_clauses_dfs();
        assert!(result.is_some());
    }

    #[test]
    fn test_intersect_clauses_dfs_not_found() {
        let tbl = BranchingTable::new(2, vec![vec![0b01], vec![0b10]]);
        let result = tbl.intersect_clauses_dfs();
        assert!(result.is_none());
    }
}
