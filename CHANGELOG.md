# Changelog

All notable changes to this crate will be documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project will adhere to [SemVer](https://semver.org/) once it
reaches 1.0.0.

## [0.1.0] — UNRELEASED

Initial public release, extracted into its own repository from the
[miso-rs](https://github.com/CodingThrust/miso-rs) workspace.

### Features
- Generic `branch_and_reduce` driver whose branching rule is chosen by
  solving a weighted set-cover problem.
- Five-trait problem interface: `BranchAndReduceProblem`, `Measure`,
  `Selector`, `TableSolver`, `Reducer`.
- `Measure` carries an associated `Output` type rather than hardcoding
  `f64`, so integer-weighted problems can stay in integer arithmetic.
- `NoReducer` re-exported at the crate root with a blanket
  `impl<P: BranchAndReduceProblem, V: num_traits::Zero> Reducer<P, V>`,
  usable directly by any problem with a `Zero` measure.
- `pub mod mock`: a self-contained `MockProblem` worked example, driving
  both `examples/mock_problem.rs` and the crate-level doc-test.
- Structured doc-comments on every public trait, struct, and function.

### Notes
- `Measure::Output`'s `From<u32>` bound admits `f64`, `u32`, `u64`,
  `i64`, but rejects `usize`, `i32`, `u16`, etc. — none of those types
  impl `From<u32>` in `core`. Newtype-wrap to provide the bound.
