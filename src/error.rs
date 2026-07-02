use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("set cover solver returned infeasible solution")]
    Infeasible,
    #[error("minimize_gamma failed to converge after {0} iterations")]
    ConvergenceFailed(usize),
    #[error("solver error: {0}")]
    SolverError(String),
}
