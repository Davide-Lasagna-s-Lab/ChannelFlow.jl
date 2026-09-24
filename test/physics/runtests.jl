# The reproducible physical validations are also the CI regression suite.
# Keep generated outputs out of the checkout during routine test runs.
ENV["CHANNEL_VALIDATION_OUTPUT"] = mktempdir()
include("../../validation/run.jl")
