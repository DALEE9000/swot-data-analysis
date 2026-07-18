"""ANN backend. A PyTorch multilayer perceptron that maps the same
stencil-flattened SWOT feature rows used by the RF backend to both SSV
components (u, v) jointly with a single shared network.

Nothing outside this package imports torch; the pipeline talks to the
ANN through the ``ANNRegressor`` / ``ANNComponentView`` wrappers, which
expose the same numpy-in / numpy-out ``predict`` interface as the RF
models."""
