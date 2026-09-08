# Replacement third-workload candidate audit

Status: `NUMINAMATH_1P5_SELECTED_PENDING_REPAIRED_PINNED_PREFLIGHT`.

This audit occurred before any NuminaMath M4 qualification, acquisition, or
causal outcome. The completed four-cell study and accepted M4 instrument are
unchanged.

DAPOMath17K remains rejected because its official artifact contains the
historically disclosed roughly 100-fold prompt duplication. DeepScaleR is also
rejected: its exact pinned 40,315-row artifact contains only 39,387 unique
problems, 612 duplicated problem values, and 171 duplicate groups with
conflicting answers. Both DeepScaleR EOS jobs were no-training preflights.

NuminaMath-1.5 is selected as the replacement candidate at immutable repository
revision `1b05109f9e5c1ad06c0663519502416c30b300f8`. The three parquet files total
531,355,727 bytes. Their locally computed SHA-256 values and sizes exactly match
the official LFS repository tree.

The first no-training preflight exposed one empty problem string at source
ordinal 279652 after the original filter retained it. Mapping failed closed
before lock creation; no training, qualification, acquisition, or causal outcome
occurred. The terminal artifact SHA-256 is
`77819e804d962581b0fcfc4927c3c39868ee8ab8d268346e3d4354c60f8b4ee5`.

The repaired repository-native valid/verifiable policy additionally requires a
nonempty, non-whitespace problem. It retains rows with nonempty answers other
than `proof` or `notfound`, non-proof question type, and both validity flags
equal to `Yes`. A column-projected audit over all three pinned shards retains
680,786 rows and finds exactly 680,786 unique nonempty problem strings: zero
duplicate groups, zero excess rows, and zero answer or solution conflicts.

The corpus uses the same problem/answer interface, math processor, and verifier
family as the accepted M4 studies. A study-local loader pins the immutable
revision and fails closed on source schema, source cardinality, filtered
cardinality, or uniqueness drift. This audit creates no EOS or training
authority.
