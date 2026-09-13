# Shuffled-time control contract

The primary shuffled-time feature is generated in `build_feature_matrices` in
`source/code/src/tomobench/evaluation/benchmark_v2_final_analysis.py`.

- The permutation unit is the complete 384-value travel-time vector of one case, not individual
  observations and not individual time positions.
- Vectors are permuted across cases separately within `train`, `validation`, and `test`; no vector
  crosses a target split boundary.
- The permutation is unrestricted, so a case may receive its own travel-time vector; no derangement
  constraint is imposed.
- Source/receiver coordinates and Euclidean distance remain those of the receiving case. Only the
  travel-time channel is copied from the selected source case. Because whole travel-time vectors are
  reassigned between independently sampled acquisition cases, the shuffled control disrupts the
  travel-time vector's association with both its generating target and its original acquisition
  geometry.
- The case/observation ordering is deterministic. Separate stable SHA-256-derived seeds are used for
  the split labels `shuffle:train`, `shuffle:validation`, and `shuffle:test`, giving seeds
  4,070,556,983; 607,745,396; and 4,143,224,087, respectively.
- After the feature matrix is constructed, the PCA-ridge coefficient map is refit on the shuffled
  training representation. The selected hyperparameters remain frozen at the Full-input-selected
  values, one PCA component and ridge alpha 1000, as recorded in
  `records/node_native/selected_ml_configuration.json`; the shuffled control is not independently
  retuned.

The comparison is therefore interpreted as a case-level association test and does not isolate a uniquely
geological timing contribution or establish causal geological information. It is not a physical inversion
method and is interpreted separately from the no-time, geometry-only, and distance-only zero-target-information
controls. The result is conditional on the predefined deterministic permutation used by the benchmark; the
target-level bootstrap interval does not include variability across alternative shuffle realizations. The
separate 100-seed sensitivity, summarized in
`records/shuffled_time/multiseed_direct_cell_overall_summary.json`, characterizes that variability.
