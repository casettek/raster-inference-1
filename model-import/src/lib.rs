#[allow(dead_code)]
#[path = "main.rs"]
mod main_impl;

pub use main_impl::{run_from_args, run_from_env};
