# Qwen3.5 V8 capability qualification

V8 is the next independent-verification experiment after the accepted V7
native qualification. V7 established preservation, official reloadability,
and parity-gated native full-sequence GDN execution. V8 asks the deliberately
different question V7 left out of scope: whether the installed, content-bound
sidecar memory is useful for retrieval.

The design is frozen in [`contract.json`](contract.json) and explained in
[`PROPOSAL.md`](PROPOSAL.md). [`source-audit.json`](source-audit.json) records
the relationship among current upstream `main`, PR 31, and the original V7
source and evidence identities.

## Status

- Design: frozen for implementation and dataset review.
- Experiment identity: not admitted; the final model, passage corpus, query
  corpus, implementation commit, and benchmark hashes have not been bound.
- Formal execution: not authorized.
- Cloud compute: not launched.

Admission remains blocked until all of the following are true:

1. the installed-memory loader verifies `lecore.json` and the sidecar SHA-256;
2. index and query encoding both use the same normalized exponential bundle,
   rather than a last-position vector on either side;
3. the query path can abstain using only the frozen calibration split;
4. a miniature end-to-end run passes on the proposed 32 GiB envelope;
5. all V7 preservation, official-interface, statistical, and native-C gates
   remain mandatory invariants; and
6. the exact inputs and source commit are contributed to and admitted by
   ilxyr before the one authorized formal execution.

The contract SHA-256 is stored in `contract.sha256`. Any change to the JSON
requires a new digest and design review. Results must live in a later stacked
evidence PR; this directory never rewrites V1-V7 evidence.
