use crate::error::FaceIdError;
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;

#[derive(Debug)]
pub struct OnnxSession {
    pub session: Session,
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
        let mut session_builder = Session::builder()?;
        if !execution_providers.is_empty() {
            session_builder = session_builder.with_execution_providers(execution_providers)?;
        }
        if let Some(opt_level) = optimization_level {
            session_builder = session_builder.with_optimization_level(opt_level)?;
        }
        if let Some(intra) = intra_threads {
            session_builder = session_builder.with_intra_threads(intra)?;
        }
        if let Some(inter) = inter_threads {
            session_builder = session_builder.with_inter_threads(inter)?;
        }
        if let Some(mem_pattern) = memory_pattern {
            session_builder = session_builder.with_memory_pattern(mem_pattern)?;
        }
        let session = session_builder.commit_from_file(path)?;

        Ok(Self { session })
    }
}
