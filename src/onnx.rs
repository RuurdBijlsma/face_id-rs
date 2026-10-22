use crate::error::FaceIdError;
use ort::environment::Environment;
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

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
        environment: Option<&Environment>,
    ) -> Result<Self, FaceIdError> {
        let mut session_builder = match environment {
            Some(environment) => Session::builder(environment)?,
            None => Session::builder(&default_environment()?)?,
        };
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

/// The environment used when the caller does not pass one with `with_environment`.
///
/// `ort` allows one environment per process, so this is built on first use and shared by every
/// session `face_id` creates. An application that also builds its own sessions should create its
/// environment first and pass it to `face_id` instead, because `ort::init().build()` fails once
/// this one exists.
fn default_environment() -> Result<Environment, FaceIdError> {
    static ENVIRONMENT: Mutex<Option<Environment>> = Mutex::new(None);
    let mut slot = ENVIRONMENT.lock().unwrap_or_else(PoisonError::into_inner);
    let environment = if let Some(environment) = slot.as_ref() {
        environment.clone()
    } else {
        let environment = ort::init().with_name("face_id").build()?;
        *slot = Some(environment.clone());
        environment
    };
    drop(slot);
    Ok(environment)
}
