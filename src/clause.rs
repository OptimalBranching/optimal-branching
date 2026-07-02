/// A conjunction of literals over bit variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Clause {
    pub mask: u64,
    pub val: u64,
}

impl Clause {
    pub fn new(mask: u64, val: u64) -> Self {
        Self {
            mask,
            val: val & mask,
        }
    }

    pub fn covered_by(&self, a: u64) -> bool {
        (a & self.mask) == self.val
    }

    /// Number of literals in the clause.
    pub fn len(&self) -> u32 {
        self.mask.count_ones()
    }

    pub fn is_empty(&self) -> bool {
        self.mask == 0
    }

    /// Check if the clause is a positive literal (single bit set, val matches mask).
    pub fn is_true_literal(&self) -> bool {
        self.mask.count_ones() == 1 && self.val == self.mask
    }

    /// Check if the clause is a negative literal (single bit set, val is zero).
    pub fn is_false_literal(&self) -> bool {
        self.mask.count_ones() == 1 && self.val == 0
    }

    pub fn literals(&self) -> Vec<Clause> {
        let mut result = Vec::new();
        let mut bit: u64 = 1;
        loop {
            if bit > self.mask || bit == 0 {
                break;
            }
            if self.mask & bit != 0 {
                result.push(Clause::new(bit, self.val & bit));
            }
            bit <<= 1;
        }
        result
    }

    pub fn bdistance(&self, b: u64) -> u32 {
        (self.val ^ (b & self.mask)).count_ones()
    }

    pub fn bdistance_clause(&self, other: &Clause) -> u32 {
        let shared_mask = self.mask & other.mask;
        ((self.val ^ other.val) & shared_mask).count_ones()
    }
}

/// Conjunction builder: combines multiple single-bit clauses into one clause.
/// Equivalent to Julia's `∧(x::Clause, xs::Clause...)`.
pub fn conjoin(clauses: &[Clause]) -> Clause {
    let mask = clauses.iter().fold(0u64, |acc, c| acc | c.mask);
    let val = clauses.iter().fold(0u64, |acc, c| acc | c.val);
    Clause::new(mask, val)
}

/// Create boolean variable clauses for bits 0..n-1.
/// Equivalent to Julia's `booleans(n)`.
pub fn booleans(n: usize) -> Vec<Clause> {
    assert!(
        n <= 64,
        "booleans(n={n}) exceeds 64-variable limit for u64-backed Clause"
    );
    (0..n)
        .map(|i| {
            let bit = 1u64 << i;
            Clause::new(bit, bit)
        })
        .collect()
}

impl std::ops::BitAnd for Clause {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        let mask = self.mask & rhs.mask;
        Clause::new(mask, self.val & rhs.val)
    }
}

impl std::ops::Not for Clause {
    type Output = Self;
    fn not(self) -> Self {
        Clause::new(self.mask, self.val ^ self.mask)
    }
}

/// Disjunctive Normal Form — OR of conjunctive clauses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DNF {
    pub clauses: Vec<Clause>,
}

impl DNF {
    pub fn covered_by(&self, a: u64) -> bool {
        self.clauses.iter().any(|c| c.covered_by(a))
    }
}

// ============================================================
// Tests ported 1:1 from Julia OptimalBranchingCore/test/bitbasis.jl
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::branching_table::{gather2, BranchingTable};
    use std::collections::HashSet;

    // Julia: bit"1110" = 0b1110 = 14
    // Julia: Clause(bit"1110", bit"0000") → Clause(14, 0 & 14) = Clause{mask:14, val:0}
    // Julia: Clause(bit"1110", bit"0001") → Clause(14, 1 & 14) = Clause{mask:14, val:0} (same!)
    // Julia: Clause(bit"1110", bit"0010") → Clause(14, 2 & 14) = Clause{mask:14, val:2}
    // Julia: Clause(bit"1100", bit"0001") → Clause(12, 1 & 12) = Clause{mask:12, val:0}

    #[test]
    fn test_clause_and_dnf() {
        let c1 = Clause::new(0b1110, 0b0000);
        let c2 = Clause::new(0b1110, 0b0001);
        let c3 = Clause::new(0b1110, 0b0010);
        let c4 = Clause::new(0b1100, 0b0001);
        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
        assert_ne!(c1, c4);

        // literals of c1: mask=0b1110, val=0b0000
        // bits 1,2,3 all have val=0 → all false literals
        let lts1 = c1.literals();
        assert_eq!(lts1.len(), 3);
        assert!(lts1.iter().all(|l| !l.is_true_literal()));
        assert!(lts1.iter().all(|l| l.is_false_literal()));

        // c2 equals c1 after masking, same literals
        let lts2 = c2.literals();
        assert_eq!(lts2.len(), 3);
        assert!(lts2.iter().all(|l| !l.is_true_literal()));
        assert!(lts2.iter().all(|l| l.is_false_literal()));

        // c3: mask=0b1110, val=0b0010 → bit1=1(true), bit2=0(false), bit3=0(false)
        let lts3 = c3.literals();
        assert_eq!(lts3.len(), 3);
        let true_lits3: Vec<bool> = lts3.iter().map(|l| l.is_true_literal()).collect();
        let false_lits3: Vec<bool> = lts3.iter().map(|l| l.is_false_literal()).collect();
        assert_eq!(true_lits3, vec![true, false, false]);
        assert_eq!(false_lits3, vec![false, true, true]);

        // c4: mask=0b1100, val=0b0000 → bit2=0(false), bit3=0(false)
        let lts4 = c4.literals();
        assert_eq!(lts4.len(), 2);
        assert!(lts4.iter().all(|l| !l.is_true_literal()));
        assert!(lts4.iter().all(|l| l.is_false_literal()));

        // c1 is not a single literal
        assert!(!c1.is_true_literal());
        assert!(!c1.is_false_literal());

        // DNF equality (Julia uses Set-based comparison)
        let dnf_1 = DNF {
            clauses: vec![c1, c2, c3],
        };
        let dnf_2 = DNF {
            clauses: vec![c1, c2, c4],
        };
        let dnf_3 = DNF {
            clauses: vec![c1, c3, c2],
        };
        let set1: HashSet<_> = dnf_1.clauses.iter().collect();
        let set2: HashSet<_> = dnf_2.clauses.iter().collect();
        let set3: HashSet<_> = dnf_3.clauses.iter().collect();
        assert_ne!(set1, set2);
        assert_eq!(set1, set3);
        assert_eq!(dnf_1.clauses.len(), 3);

        // bdistance
        let cstr = 0b0011u64;
        assert_eq!(c2.bdistance_clause(&c3), 1);
        assert_eq!(c2.bdistance(cstr), 1);
    }

    #[test]
    #[should_panic(expected = "exceeds 64-variable limit")]
    fn test_booleans_panics_over_64() {
        booleans(65);
    }

    // Ported from Julia bitbasis.jl "gather2" testset
    #[test]
    fn test_gather2() {
        // Julia: mask = bmask(INT, 1:5) = 31
        // v1 = 0b00010 = 2, v2 = 0b01001 = 9
        let c1 = Clause::new(31, 2);
        let c2 = Clause::new(31, 9);
        let c3 = gather2(5, &c1, &c2);
        // Expected: Clause(0b10100=20, 0)
        assert_eq!(c3, Clause::new(20, 0));
        assert_eq!(c3.len(), 2);
    }

    // Ported from Julia bitbasis.jl "satellite" testset
    // Julia table: BranchingTable(5, [
    //   [SEV(2,[0,0,1,0,0]), SEV(2,[0,1,0,0,0])],  → [4, 2]
    //   [SEV(2,[1,0,0,1,0])],                        → [9]
    //   [SEV(2,[0,0,1,0,1])]                          → [20]
    // ])
    #[test]
    fn test_satellite() {
        let tbl = BranchingTable::new(5, vec![vec![4, 2], vec![9], vec![20]]);
        let vars = booleans(5);
        let (a, b, c, d, e) = (vars[0], vars[1], vars[2], vars[3], vars[4]);

        // !covered_by(tbl, DNF(a ∧ ¬b))
        assert!(!tbl.covered_by(&DNF {
            clauses: vec![conjoin(&[a, !b])]
        }));

        // covered_by(tbl, DNF(a ∧ ¬b ∧ d ∧ ¬e, ¬a ∧ ¬b ∧ c ∧ ¬d))
        let c1 = conjoin(&[a, !b, d, !e]);
        let c2 = conjoin(&[!a, !b, c, !d]);
        assert!(tbl.covered_by(&DNF {
            clauses: vec![c1, c2]
        }));

        // !covered_by(tbl, DNF(a ∧ ¬b ∧ d ∧ ¬e, ¬a ∧ ¬b ∧ c ∧ ¬d ∧ e))
        let c2_e = conjoin(&[!a, !b, c, !d, e]);
        assert!(!tbl.covered_by(&DNF {
            clauses: vec![c1, c2_e]
        }));

        // covered_by(tbl, DNF(a ∧ ¬b ∧ d ∧ ¬e, ¬a ∧ ¬b ∧ c ∧ ¬d ∧ e, ¬a ∧ b ∧ ¬c ∧ ¬d ∧ ¬e))
        let c3 = conjoin(&[!a, b, !c, !d, !e]);
        assert!(tbl.covered_by(&DNF {
            clauses: vec![c1, c2_e, c3]
        }));
    }
}
