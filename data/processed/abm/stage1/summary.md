# STAGE1 Summary

## Mechanisms Present
- latent anchor with Gaussian drift plus occasional jumps
- three trader classes only: informed, noise, herding
- base participation with deterministic deadline ramp
- single reduced-form nonlinear impact rule with static effective depth
- hard cash, inventory, position, and per-step order caps
- explicit no-move threshold

## Empirical Targets Matched
- volatility clustering

## Targets Still Missing or Weak
- heavy tails
- late life volatility

## Calibration Snapshot
- total score: 0.7014
- tail error: 1.2864
- acf error: 0.3653
- deadline error: 0.4122
- ratio error: 0.5677
- monotonic penalty: 0.0000

## Next Stage Decision
- justified: yes
- reason: Stage 2 is justified because the activity x deadline sign pattern is not present by construction in stage 1.