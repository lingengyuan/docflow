enum SourceState {
    Ready,
    NeedsReview,
}

fn rust_enum_parser_evidence(state: SourceState) -> &'static str {
    match state {
        SourceState::Ready => "ready",
        SourceState::NeedsReview => "needs review",
    }
}
