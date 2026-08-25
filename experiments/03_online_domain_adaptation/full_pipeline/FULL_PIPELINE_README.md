# Full label-blind online-style evaluation

`full_label_blind_online_evaluation.py` combines the tuned blind EMG burst
detector with the frozen epoch-147 Meta adapter checkpoint.

## Inference protocol

1. Reconstruct the earlier session as one continuous EMG stream.
2. Calibrate and run the burst detector using EMG only.
3. Center a one-second window on each predicted burst.
4. Select model output bins using the predicted burst boundaries.
5. Classify with the frozen model and save the decision.
6. Load final auto-relabeled `trainKin` only after all predictions are fixed.
7. Match and score detected events against the reference events.

No original prompt labels, label-centered windows, label-selected active bins,
optimizer steps, or artifact masks are used during inference.

## Current benchmark

- 99/99 reference bursts detected
- 9 false detections
- 91.7% detection precision
- 100% detection recall
- 75.8% gesture accuracy conditional on a matched detection
- 75.8% end-to-end correct recall
- 69.4% end-to-end correct precision including false detections
- 90% gesture accuracy on the prior session's test subset

## Interpretation

This is a label-blind, buffered online-style replay. It is not yet a final
untouched detector test because the detector merge setting was selected after
examining this session. The one-second preprocessing also uses zero-phase
filtering after the complete window is buffered; a live sample-by-sample
implementation should replace that stage with a causal stateful filter.
