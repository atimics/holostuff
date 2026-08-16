# RFC: Qwen3.5 V8 installed-memory capability qualification

- Status: design frozen; implementation and input identities pending
- Verification spine: PR 31
- Prior result: V7 accepted — preservation and native full-sequence C qualified
- Formal attempts authorized by this document: one, after ilxyr admission

## Decision

V8 does not repeat V7 and does not reopen its accepted result. It evaluates one
new claim:

> Does the content-addressed memory installed beside Qwen3.5-0.8B retrieve the
> correct grounded passage from unseen paraphrased queries, outperform a
> lexical control on the same queries, and abstain on queries with no answer in
> the installed corpus, while preserving every V7 safety invariant?

The treatment is leCore's installed sidecar memory with a shared bundled-state
address encoder for indexing and querying. The primary control is a frozen
BM25 implementation over the same passages. A seed-zero shuffled index is a
negative control. The no-sidecar path must refuse rather than invent a result.

V8 qualifies *host-mediated installed retrieval*. It does not claim that the
model autonomously decides when to search, that retrieved text is automatically
injected into generation, or that model intelligence improves.

## Why V8 is needed

V7 deliberately excluded usefulness of installed memory, routing, registers,
and related capabilities. Its paired language-model evaluation also disabled
the sidecar, correctly proving that the safe installation did not perturb
ordinary logits but providing no evidence that the sidecar helped a user.

The present installer writes one normalized last-position vector per passage.
Its retained measurement reports 9/18 correct partial-cue retrievals. The
separate memory implementation records why that representation is unsuitable:
the last state reflects recent tokens, whereas a normalized exponential bundle
over positions retrieved 62/64 at 128 dimensions in its controlled fixture.
V8 therefore refuses admission until index and query encoders share the same
bundled-state definition and the emitted artifact declares that definition.

## Verification base and frozen history

The design starts from PR 31 head
`3091ae40836373992e45761c80c872ddb18d5fd5`. That head already contains
upstream `main` commit `9c1fdd50af4ac18c07d0612ac39dfac144c3f6d3`;
rebasing was consequently a no-op and would only have rewritten evidence
history.

V7's original source is
`cb3b1d2ac71c183bf9307ca7145a2a619ff30c30`. Its stable patch ID equals that
of rebased source-equivalent commit
`d140c110ca7c1e12c3e6457225f37634d209ecac`. The committed V7 result tree and
the V6/V7 PDFs remain byte-identical. Exact identities are recorded in
`source-audit.json`; the original commit remains available by full GitHub SHA
and the evidence is independently reconstructable from its permanent bundle.

Once the final V8 source is admitted, no branch rebase, threshold change, data
replacement, or code change is permitted until the terminal outcome is
published. PR 31 may resume tracking upstream only after that publication.

## Subject and inputs

- Model: public `Qwen/Qwen3.5-0.8B`, bound by a complete file manifest.
- Installation corpus: a redistributable corpus, distinct from every query
  authoring source and bound by SHA-256.
- Passage set: exactly 200 deterministic passages selected before query
  evaluation.
- Answerable evaluation queries: exactly 160 independently authored
  paraphrases, each mapped to one passage identifier.
- Abstention calibration queries: exactly 40 queries with no answer in the
  passage set. They select one score threshold before formal evaluation.
- Unanswerable evaluation queries: exactly 40 additional queries, disjoint from
  calibration, with no answer in the passage set.

The 240 formal/calibration query rows are canonical JSONL. Each row contains a
stable ID, split, query, answerability, and expected passage SHA-256 when
answerable. The generator must reject duplicate IDs, duplicate query text,
unknown passage hashes, split-count drift, or an answerable query that contains
a normalized 24-character substring copied from its expected passage. Query
authors may see the factual content needed to write a paraphrase but may not
inspect retrieval rankings or scores.

## Addressing contract

Both index and query paths use the final runtime layer and the identical
address function:

1. collect every hidden state for at most 200 tokens;
2. apply the normalized exponential accumulator with decay `0.99`;
3. subtract the mean address computed only from the 200 indexed passages;
4. L2-normalize with epsilon `1e-30`; and
5. rank by cosine similarity with deterministic passage-ID tie-breaking.

The `.npz` sidecar records schema name, passage IDs and hashes, addresses,
mean, decay, layer, token limit, dtype, and encoder name. `lecore.json` binds
the complete sidecar bytes by SHA-256. Loading or searching fails closed on a
digest mismatch, a missing field, non-finite data, dimension mismatch, or an
unknown encoder.

Legacy V7 last-position sidecars are preserved as evidence but are not valid
V8 treatment artifacts.

## Controls

### Lexical control

BM25 sees the same passage text and query rows. Tokenization, `k1=1.2`,
`b=0.75`, lowercasing, Unicode normalization, punctuation policy, and stable
tie-breaking are frozen in the runner policy. It receives no model states.

### Shuffled negative control

The treatment address matrix is paired with a seed-zero permutation of passage
identities. Its purpose is to catch accidental answer leakage or evaluation
wiring that ignores the ranked identity.

### No-sidecar control

Removing the sidecar or its manifest binding must produce a typed refusal. It
must never silently return BM25 output or an arbitrary passage under the
treatment label.

## Metrics

Primary metrics on the 160 answerable queries are treatment Recall@1,
Recall@5, mean reciprocal rank, and paired treatment-minus-BM25 Recall@1.
The paired difference uses a two-sided 95% percentile bootstrap with 10,000
seed-zero resamples over query rows.

Abstention calibration selects the highest similarity threshold that keeps
false positives at or below 5% on the 40 calibration queries; ties choose the
more conservative threshold. The frozen threshold is then applied without
change to the 40 unanswerable evaluation queries. Report unanswerable false
positive rate, answerable abstention rate, and coverage.

Operational metrics are per-query p50/p95 latency, index-build wall time,
installer wall time, peak RSS, emitted checkpoint size, sidecar size, native C
provenance, and total compute estimate. They are evidence, not substitutes for
the retrieval decision.

## Acceptance rule

An `accepted` outcome requires every conjunct below:

1. all V7 preservation, official reload, text, vision, statistical, source,
   spectral-disabled, and native full-sequence C gates pass;
2. exactly 160 answerable and 40 held-out unanswerable evaluation rows complete;
3. treatment Recall@1 is at least `0.80`;
4. treatment Recall@5 is at least `0.95`;
5. the lower endpoint of the paired 95% bootstrap interval for
   treatment-minus-BM25 Recall@1 is strictly greater than `0.0`;
6. held-out unanswerable false-positive rate is at most `0.05`;
7. shuffled-control Recall@1 is at most `0.10`;
8. the treatment sidecar passes its content, schema, and encoder-integrity
   checks; and
9. the complete run finishes on a 32 GiB instance with peak RSS at or below
   `28,672 MiB`, leaving at least 4 GiB for the system envelope.

Failure of a scientific gate after a valid metric envelope is `rejected`.
Dependency failure, timeout, malformed output, integrity refusal before the
metric envelope, or incomplete evaluation is `execution_failure`. All three
outcomes are published. Early acceptance and retries under the same identity
are forbidden.

## Performance envelope

V7 observed 19,503.84 MiB peak RSS. Its frozen right-sizing report projected
that the full run requires 27,786,149,888 bytes including headroom and reserve,
leaving about 6.1 GiB on a 32 GiB class. V8 uses 32 GiB only after a miniature
end-to-end rehearsal confirms the complete installer, two-runtime safety gate,
and retrieval evaluation fit.

V7 installation took 8,056.60 seconds, about 93% of terminal runtime. V8 must
profile index construction separately and report passages per second. Installer
speed is not a scientific GO gate in this first capability experiment, so an
optimization cannot rescue a failed retrieval result.

The formal timeout remains six hours and estimated compute must remain under
USD 10. No spot instance is used for the one formal attempt.

## Explicit exclusions

- native cached-step generation qualification;
- spectral filtering or weight-resident metadata;
- autonomous search triggering;
- generation-quality or intelligence-improvement claims;
- models other than the frozen Qwen3.5-0.8B subject;
- threshold tuning after admission; and
- promotion of the sidecar to a default-on production feature.

## Publication

The ilxyr project binds the source, model manifest, passage corpus, query JSONL,
runner policy, benchmark, dependency lock, sidecar schema, and every threshold.
The result publishes the terminal envelope, raw per-query ranks and scores,
bootstrap draws or their deterministic reconstruction inputs, runtime logs,
ledger, and permanent receipt. Model weights, credentials, and temporary cloud
paths remain excluded.
