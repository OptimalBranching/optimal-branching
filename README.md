# optimal-branching

Generic branch-and-reduce framework, with the branching rule chosen by
solving a weighted set-cover problem. Rust port of the Julia
[`OptimalBranching.jl`](https://github.com/OptimalBranching/OptimalBranching.jl).

Paper: *Automated discovery of optimal branching rules for the
branch-and-bound algorithm*, arXiv [2412.07685](https://arxiv.org/abs/2412.07685).

## Status

`0.1.x`. The public API is stable enough for downstream experimentation
but may evolve before 1.0; see [CHANGELOG.md](./CHANGELOG.md).

## Install

```toml
[dependencies]
optimal-branching = "0.1"
```

## Hello world

```rust
use optimal_branching::{
    branch_and_reduce, BranchingStrategy, MaxSize, NoReducer,
    mock::{MockProblem, MockTableSolver, NumOfVariables, RandomSelector},
    solver::IPSolver,
};

let problem = MockProblem { optimal: vec![true, false, true] };
let strategy = BranchingStrategy::new(
    MockTableSolver { n: 16, p: 0.3, seed: 42 },
    IPSolver::default(),
    RandomSelector { n: 2, seed: 42 },
    NumOfVariables,
    NoReducer,
);
let answer: MaxSize = branch_and_reduce(problem, &strategy).unwrap();
```

Run the example end-to-end:

```bash
cargo run --release -p optimal-branching --example mock_problem
```

## Implementing your own problem

You implement five traits:

- `BranchAndReduceProblem` — your problem type (`is_empty`, `apply_branch`).
- `Measure` — a number that captures problem hardness.
- `Selector` — picks the next variables to branch on.
- `TableSolver` — enumerates legal assignments for a region.
- `Reducer` — (optional) local rewriting. Use `NoReducer` to skip.

The `mock` module ships a complete worked example; copy and edit it.

## License

MIT.
