"""AlphaEvolve-style evolutionary code search for the SWOTxAI ANN backend.

An LLM (Claude API) proposes code mutations to a candidate training module;
each candidate is trained locally in a sandboxed subprocess and scored against
HFR ground truth on a temporally held-out split. Everything lives outside the
shipped pipeline: candidates are generated files under experiments/evolve/ and
the models/ann package is only ever read as the seed, never modified.

Heavy imports (torch, anthropic) stay inside the submodules that need them.
"""
