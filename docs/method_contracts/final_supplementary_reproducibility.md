# Implementation-level method contracts

This record documents the implementation-level contracts of the reported analysis at a level of
detail below that of the Supplementary Material. It describes the production code, stored
artifacts, and released outputs of the reported results.

## 1. Forward endpoint contract

The pinned finite-grid forward endpoint contract is documented in
`records/forward_solver_endpoint_contract.md`. In brief, the production path is
node-centered ttcrpy==1.4.2 FSM with `tt_from_rp=False`, `interp_vel=False`, WENO
enabled, `eps=1e-5`, `maxit=100`, and `translate_grid=False`. Source and receiver
arrays are passed at their saved continuous coordinates after only project-side
finiteness and closed-domain checks. Off-node source initialization is a local
source-containing-cell stencil initialized from source-to-node distance and
nodal slowness; it is not nearest-node snapping. Non-raypath receiver
evaluation reads the converged nodal FSM field, using direct nodal values,
linear edge interpolation, bilinear face interpolation, or trilinear
interpolation of eight surrounding nodal values as applicable.

## 2. Reference-ray linear solver and smoothing operator

The implementation is `solve_fixed_ray_case` and `_solve_pcg` in
`source/code/src/tomobench/evaluation/benchmark_v2_final_analysis.py`. The selected
configuration has nonzero smoothing (damping 0.001, smoothing 10.0), so the
selected systems use the project preconditioned conjugate-gradient (PCG)
routine rather than the zero-smoothing `numpy.linalg.solve` dual branch.

The PCG operator is the fixed-ray normal operator plus damping and smoothing.
The preconditioner is diagonal Jacobi with entries
`damping + sum(path_length_km^2) + smoothing*degree`, floored at `1e-12`.
The initial solution is zero. The routine uses
`rhs_norm=max(||rhs||,1e-15)`. It returns success at initialization when
`sqrt(max(r^T M^-1 r,0))/rhs_norm <= 1e-8`, or after an iteration when
`||r||/rhs_norm <= 1e-8`; it fails on a nonpositive/nonfinite search
denominator and otherwise returns failure at the 2,000-iteration cap. The
selected-test diagnostics contain 38 rows with `pcg_converged=True`.
Thus "all 38 test solves were solved successfully" means that each of the
38 selected reference-ray test linear systems returned the successful PCG
status under these checks.

The smoothing operator is implicit `L^T L` on the target-cell slowness vector;
`L` is not materialized. The vector is reshaped as `(nz, ny, nx)` and flattened
back in C order, so x is the fastest axis. Its conceptual rows are raw
nearest-neighbor differences in x, then y, then z order:
`value[neighbor] - value[current]`. Differences are not divided by the
2.5-km cell spacing. All three axes have the same unit directional weight.
The operator is nonperiodic: only in-domain neighbors contribute, so boundary
cells have one-sided incidence and no wrap-around. The same convention is used
by the degree term in the Jacobi preconditioner.

The reference-ray validation candidates are damping
`{1e-5,1e-4,1e-3,1e-2,1e-1,1,10}` and smoothing `{0,0.1,1,10,100,1000}`.
The selected values are validation-optimal within that predefined finite grid;
boundary selection is not interpreted as global hyperparameter optimality. A
smaller four-by-four candidate grid appears in the earlier secondary-diagnostic
records under `records/secondary_diagnostics/` and is not the selection contract
for the reported results.

## 3. Figure 4 selection contract

The selection is produced by
`source/code/src/tomobench/evaluation/final_evidence_pass.py`. The population
is the 38 held-out test cases. For each case, the common selection metric is the
Full-input PCA-ridge direct analytic cell-center RMSE over all target cells.
Cases are sorted ascending by that metric and then by `target_hash` as the
deterministic tie-break. The selected order positions are zero-based
`len(test_cases)//4` and `(3*len(test_cases))//4`, namely indices 9 and 28 for 38
cases (the 10th and 29th sorted cases). Exactly two cases are selected; no
random seed is used.

The released selection record is `records/figures/figure_4_selection.json`, and it
accompanies the true, predicted, and error fields for the two direct-cell-selected
cases. Earlier eight-corner selection records are not used for this figure.

## 4. RNG and seed convention

The production seed namespace is `base_seed=30260829`, equal to the pilot
base seed 20260829 plus 10,000,000. Targets are traversed in the configured
family order layered, block_anomaly, faulted, salt_dome, dyke_intrusion,
with a one-based global target index. For family index `family_index` and
duplicate-resolution attempt, the stored seeds are:

- `geology_seed = base_seed + 1,000,000 + target_index*1,000 + family_index*100 + attempt`;
- `station_seed = base_seed + 2,000,000 + target_index`; and
- `earthquake_seed = base_seed + 3,000,000 + target_index`.

These derived seeds are stored in the target registry and target metadata.
Geological parameter sampling uses a local `numpy.random.default_rng` under
the recorded NumPy 1.26.4 environment (PCG64). Station and earthquake
generation use local Python `random.Random` instances (the CPython
Mersenne-Twister implementation) with their respective stored seeds. No
global RNG state is seeded for these generators. Reproducing the stored
draws therefore requires the recorded NumPy version, not only the recorded
seeds.

The shuffled-time control uses a local `numpy.random.default_rng` with the
stable seed
`int.from_bytes(sha256(label.encode("utf-8")).digest()[:8],"little") %
(2**32-1)` for `label=shuffle:train`, `shuffle:validation`, and
`shuffle:test`. The resulting stored split seeds are 4,070,556,983;
607,745,396; and 4,143,224,087. Each split has one deterministic,
unrestricted whole-vector permutation; no additional realization is used by
the frozen ablation, and the separate 100-seed sensitivity is recorded under
`records/shuffled_time/`.

For timing noise, zero noise uses the stored repetition tuple `(0,)` and
does not draw a perturbation. Nonzero levels use repetitions 1 through 10.
Each case-level draw uses a local NumPy generator seeded from the stable
label `noise:{round(noise_level*1000)}:{repetition}:{target_id}` and adds
independent zero-mean Gaussian values to the 384-entry travel-time channel.

## 5. Target-hash serialization

The canonical source target is the flattened target-grid `p_velocity_km_per_s`
vector with 41 x 41 x 13 nodes, length 21,853, in the project's x-fastest
C-order convention. The serializer casts to little-endian IEEE-754 float64
(`<f8`), requires a one-dimensional finite nonempty vector, and makes it
contiguous in C order. It performs no rounding, text formatting, or further
casting.

The exact SHA-256 input is the ASCII header
`target_velocity_vector_sha256_v1\0dtype=<f8\0length=21853\0`
concatenated with the raw contiguous little-endian float64 bytes in C order.
The digest is emitted as lowercase hexadecimal. The 250 target hashes in the
released registry are produced by this procedure.

## 6. Paired win/loss/tie rule

Paired comparisons use raw full-precision floating-point deltas before display
rounding. With `TIE_TOLERANCE=1e-12`, a win for the first method is
`delta < -1e-12`, a loss is `delta > 1e-12`, and a tie is
`abs(delta) <= 1e-12`. The Table 3 counts, including zero ties,
are based on this rule, not on rounded display values.

## 7. Generator-defined structural masks

Cell centers are midpoints of adjacent target-grid coordinates. The
`meshgrid(indexing="xy")` result is transposed to `(z,y,x)` and flattened in
C order, giving x-fastest membership. No boundary or buffer cells are
excluded.

For layered targets, the layer index is
`searchsorted(depth_boundaries[1:-1], z_center, side="right")`; the layer
masks cover every cell center. For body families, background is the strict
Boolean complement of the body mask:

- block anomaly: `abs(x-cx)<=sx/2`, `abs(y-cy)<=sy/2`, and `abs(z-cz)<=sz/2`;
- salt dome: `((x-cx)/rx)^2 + ((y-cy)/ry)^2 + ((z-cz)/rz)^2 <= 1`;
- dyke: `abs((x-cx)cos(strike)+(y-cy)sin(strike)) <= length/2`,
  `abs(-(x-cx)sin(strike)+(y-cy)cos(strike)) <= width/2`, and
  `top_depth <= z <= bottom_depth`.

For faulted targets, the analytic signed fault-plane offset is
evaluated at each cell center using the stored fault position, strike, dip,
dip direction, and positive-side convention. The positive-side mask is
`signed >= 0` for `greater_equal` and `signed <= 0` otherwise; the negative
side is its strict Boolean complement. These are generator geometry masks,
not predicted-structure detectors.

## 8. Scope of this record

The endpoint, solver, smoothing, selection, RNG, hash, tie, and mask
contracts above are static descriptions of the implementation and artifacts that
produced the reported results. They are documentation only and do not define a
new analysis.
