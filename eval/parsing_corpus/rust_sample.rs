fn stable_chunk_id(file_id: &str, chunk_index: usize) -> String {
    format!("{}:{}", file_id, chunk_index)
}

// rust parser evidence keeps code text searchable.
