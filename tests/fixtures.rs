use optimal_branching::{
    minimize_gamma, BranchingTable, CandidateClause, Clause, IPSolver, SetCoverSolver,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

const FIXTURE_DIR: &str = "tests/data/jl";

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIR)
        .join(name)
}

/// Read fixture file, returning None if it doesn't exist.
/// Tests should skip (return early) when fixtures are not generated.
fn read_fixture(name: &str) -> Option<String> {
    let path = fixture_path(name);
    if !path.exists() {
        eprintln!("Skipping: fixture {name} not found. Run `julia scripts/generate_fixtures.jl` to generate.");
        return None;
    }
    Some(std::fs::read_to_string(path).unwrap())
}

#[derive(Deserialize)]
struct ComplexityBvFixture {
    name: String,
    branching_vector: Vec<f64>,
    gamma: f64,
}

#[test]
fn test_complexity_bv_fixtures() {
    let Some(data) = read_fixture("complexity_bv.json") else {
        return;
    };
    let fixtures: Vec<ComplexityBvFixture> = serde_json::from_str(&data).unwrap();
    assert!(!fixtures.is_empty(), "no complexity_bv fixtures found");

    for f in &fixtures {
        let actual = optimal_branching::complexity_bv(&f.branching_vector);
        assert!(
            (actual - f.gamma).abs() < 1e-10,
            "complexity_bv mismatch for '{}': expected {}, got {}",
            f.name,
            f.gamma,
            actual
        );
    }
}

#[derive(Deserialize)]
struct ClauseInput {
    mask: u64,
    val: u64,
}

#[derive(Deserialize)]
struct CoveredByCheck {
    input: u64,
    output: bool,
}

#[derive(Deserialize)]
struct ClauseOperations {
    len: u32,
    is_true_literal: bool,
    is_false_literal: bool,
    literals: Vec<ClauseInput>,
    covered_by: Vec<CoveredByCheck>,
}

#[derive(Deserialize)]
struct ClauseOpsFixture {
    name: String,
    clause: ClauseInput,
    operations: ClauseOperations,
}

#[test]
fn test_clause_ops_fixtures() {
    let Some(data) = read_fixture("clause_ops.json") else {
        return;
    };
    let fixtures: Vec<ClauseOpsFixture> = serde_json::from_str(&data).unwrap();
    assert!(!fixtures.is_empty(), "no clause_ops fixtures found");

    for f in &fixtures {
        let clause = optimal_branching::Clause::new(f.clause.mask, f.clause.val);

        assert_eq!(clause.len(), f.operations.len, "'{}' len mismatch", f.name);
        assert_eq!(
            clause.is_true_literal(),
            f.operations.is_true_literal,
            "'{}' is_true_literal mismatch",
            f.name
        );
        assert_eq!(
            clause.is_false_literal(),
            f.operations.is_false_literal,
            "'{}' is_false_literal mismatch",
            f.name
        );

        // Check literals (as sets of (mask, val))
        let actual_literals: std::collections::HashSet<(u64, u64)> =
            clause.literals().iter().map(|l| (l.mask, l.val)).collect();
        let expected_literals: std::collections::HashSet<(u64, u64)> = f
            .operations
            .literals
            .iter()
            .map(|l| (l.mask, l.val))
            .collect();
        assert_eq!(
            actual_literals, expected_literals,
            "'{}' literals mismatch",
            f.name
        );

        // Check covered_by
        for check in &f.operations.covered_by {
            assert_eq!(
                clause.covered_by(check.input),
                check.output,
                "'{}' covered_by({}) mismatch",
                f.name,
                check.input
            );
        }
    }
}

#[derive(Deserialize)]
struct BranchingTableFixture {
    name: String,
    bit_length: usize,
    table: Vec<Vec<u64>>,
    candidate_clauses: Vec<ClauseInput>,
    covered_items: Vec<CoveredItemsEntry>,
}

#[derive(Deserialize)]
struct CoveredItemsEntry {
    clause: ClauseInput,
    items: Vec<usize>,
}

#[test]
fn test_candidate_clauses_fixtures() {
    let Some(data) = read_fixture("branching_tables.json") else {
        return;
    };
    let fixtures: Vec<BranchingTableFixture> = serde_json::from_str(&data).unwrap();
    assert!(!fixtures.is_empty());

    for f in &fixtures {
        let table = optimal_branching::BranchingTable::new(f.bit_length, f.table.clone());
        let actual = table.candidate_clauses();
        let actual_set: HashSet<(u64, u64)> = actual
            .iter()
            .map(|c| (c.clause.mask, c.clause.val))
            .collect();
        let expected_set: HashSet<(u64, u64)> = f
            .candidate_clauses
            .iter()
            .map(|c| (c.mask, c.val))
            .collect();
        assert_eq!(
            actual_set, expected_set,
            "'{}' candidate_clauses mismatch.\nActual count: {}, Expected count: {}\nMissing from actual: {:?}\nExtra in actual: {:?}",
            f.name, actual_set.len(), expected_set.len(),
            expected_set.difference(&actual_set).collect::<Vec<_>>(),
            actual_set.difference(&expected_set).collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_covered_items_fixtures() {
    let Some(data) = read_fixture("branching_tables.json") else {
        return;
    };
    let fixtures: Vec<BranchingTableFixture> = serde_json::from_str(&data).unwrap();
    assert!(!fixtures.is_empty());

    for f in &fixtures {
        let table = optimal_branching::BranchingTable::new(f.bit_length, f.table.clone());
        for entry in &f.covered_items {
            let clause = optimal_branching::Clause::new(entry.clause.mask, entry.clause.val);
            let mut actual = table.covered_items(&clause);
            actual.sort();
            let mut expected = entry.items.clone();
            expected.sort();
            assert_eq!(
                actual, expected,
                "'{}' covered_items mismatch for clause (mask={}, val={})",
                f.name, entry.clause.mask, entry.clause.val
            );
        }
    }
}

#[derive(Deserialize)]
struct SetCoverFixture {
    name: String,
    weights: Vec<f64>,
    subsets: Vec<Vec<usize>>,
    num_items: usize,
    expected_total_weight: f64,
}

#[test]
fn test_set_cover_fixtures() {
    let Some(data) = read_fixture("set_cover.json") else {
        return;
    };
    let fixtures: Vec<SetCoverFixture> = serde_json::from_str(&data).unwrap();
    assert!(!fixtures.is_empty());

    let solver = IPSolver::default();
    for f in &fixtures {
        let result = solver.solve(&f.subsets, f.num_items, &f.weights).unwrap();

        // Verify all items are covered
        let mut covered = std::collections::HashSet::new();
        for &i in &result {
            for &g in &f.subsets[i] {
                covered.insert(g);
            }
        }
        for item in 0..f.num_items {
            assert!(
                covered.contains(&item),
                "'{}' item {} not covered",
                f.name,
                item
            );
        }

        // Verify total weight matches expected
        let total_weight: f64 = result.iter().map(|&i| f.weights[i]).sum();
        assert!(
            (total_weight - f.expected_total_weight).abs() < 1e-6,
            "'{}' total weight mismatch: expected {}, got {}",
            f.name,
            f.expected_total_weight,
            total_weight
        );
    }
}

#[derive(Deserialize)]
struct MinimizeGammaOutput {
    gamma: f64,
    optimal_rule: Vec<ClauseInput>,
    branching_vector: Vec<f64>,
}

#[derive(Deserialize)]
struct MinimizeGammaFixture {
    name: String,
    bit_length: usize,
    table: Vec<Vec<u64>>,
    candidates: Vec<ClauseInput>,
    delta_rho: Vec<f64>,
    output: MinimizeGammaOutput,
}

#[test]
fn test_minimize_gamma_fixtures() {
    let Some(data) = read_fixture("minimize_gamma.json") else {
        return;
    };
    let fixtures: Vec<MinimizeGammaFixture> = serde_json::from_str(&data).unwrap();
    assert!(!fixtures.is_empty());

    let solver = IPSolver::default();
    for f in &fixtures {
        let table = BranchingTable::new(f.bit_length, f.table.clone());

        // Reconstruct CandidateClause by computing covered_items from the table
        let candidates: Vec<CandidateClause> = f
            .candidates
            .iter()
            .map(|c| {
                let clause = Clause::new(c.mask, c.val);
                let covered_items = table.covered_items(&clause);
                CandidateClause {
                    clause,
                    covered_items,
                }
            })
            .collect();

        let result = minimize_gamma(&table, &candidates, &f.delta_rho, &solver).unwrap();

        // Compare gamma within tolerance
        assert!(
            (result.gamma - f.output.gamma).abs() < 1e-4,
            "'{}' gamma mismatch: expected {}, got {}",
            f.name,
            f.output.gamma,
            result.gamma
        );

        // Compare optimal_rule as set of clauses
        let actual_rule: HashSet<(u64, u64)> = result
            .optimal_rule
            .clauses
            .iter()
            .map(|c| (c.mask, c.val))
            .collect();
        let expected_rule: HashSet<(u64, u64)> = f
            .output
            .optimal_rule
            .iter()
            .map(|c| (c.mask, c.val))
            .collect();
        assert_eq!(
            actual_rule, expected_rule,
            "'{}' optimal_rule mismatch",
            f.name
        );

        // Compare branching_vector (sorted, within tolerance)
        let mut actual_bv = result.branching_vector.clone();
        actual_bv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected_bv = f.output.branching_vector.clone();
        expected_bv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            actual_bv.len(),
            expected_bv.len(),
            "'{}' branching_vector length mismatch",
            f.name
        );
        for (a, e) in actual_bv.iter().zip(expected_bv.iter()) {
            assert!(
                (a - e).abs() < 1e-6,
                "'{}' branching_vector mismatch: expected {:?}, got {:?}",
                f.name,
                expected_bv,
                actual_bv
            );
        }
    }
}
