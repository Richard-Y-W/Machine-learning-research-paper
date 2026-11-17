"""Model evolution by weight-space programming (WSP).

LLM-authored weight edits act as mutations; offspring inherit their parent's
cumulative edit list (the genome); selection uses development data only.
See docs/evolution-protocol.md.
"""
