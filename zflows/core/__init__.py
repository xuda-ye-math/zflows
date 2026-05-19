"""Internal machinery for zflows flows.

Organised after zuko's layout (transforms, MLPs, ODE solver) with two
deliberate divergences:
    1. there is no `context` / conditional-on-c plumbing anywhere;
    2. MonotonicRQSTransform / CircularShiftTransform accept a per-coord
       `bound` tensor so spline knots can span [-bound_i, bound_i] without
       an affine scaling sandwich.

The public flow API lives in zflows/flow.py and re-exports
`ComposedTransform` for downstream code.
"""
