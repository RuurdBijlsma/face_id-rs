use crate::error::FaceIdError;
use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;

#[derive(Debug)]
pub struct OnnxSession {
    pub session: Session,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub optimization_level: Option<GraphOptimizationLevel>,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
    pub memory_pattern: Option<bool>,
}

impl OnnxSession {
    pub fn new(
        path: impl AsRef<Path>,
        execution_providers: &[ExecutionProviderDispatch],
        optimization_level: Option<GraphOptimizationLevel>,
        intra_threads: Option<usize>,
        inter_threads: Option<usize>,
        memory_pattern: Option<bool>,
    ) -> Result<Self, FaceIdError> {
        let eps = execution_providers;
        let mut session_builder = Session::builder()?;
        if !eps.is_empty() {
            session_builder = session_builder.with_execution_providers(eps)?;
        }
        if let Some(optimization_level) = optimization_level {
            session_builder = session_builder.with_optimization_level(optimization_level)?;
        }
        if let Some(intra_threads) = intra_threads {
            session_builder = session_builder.with_intra_threads(intra_threads)?;
        }
        if let Some(inter_threads) = inter_threads {
            session_builder = session_builder.with_inter_threads(inter_threads)?;
        }
        if let Some(memory_pattern) = memory_pattern {
            session_builder = session_builder.with_memory_pattern(memory_pattern)?;
        }
        let session = session_builder.commit_from_file(path)?;

        Ok(Self {
            session,
            execution_providers: eps.to_vec(),
            optimization_level,
            intra_threads,
            inter_threads,
            memory_pattern,
        })
    }
}
