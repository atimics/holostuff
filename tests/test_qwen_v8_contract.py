"""Static integrity gates for the frozen Qwen V8 capability design."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "experiments" / "qwen35_capability_v8"
CONTRACT = V8 / "contract.json"
AUDIT = V8 / "source-audit.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_v8_contract_freezes_one_new_capability_question():
    policy = load(CONTRACT)
    assert policy["schema"] == "lecore.qwen35.capability-policy.v1"
    assert policy["experiment_version"] == 8
    assert policy["status"] == "design_frozen_inputs_pending"
    assert policy["acceptance"]["formal_attempts"] == 1
    assert policy["acceptance"]["early_acceptance_allowed"] is False
    assert policy["execution"]["cloud_execution_authorized"] is False
    assert "generation_quality_claims" in policy["exclusions"]


def test_v8_query_splits_and_thresholds_are_closed():
    policy = load(CONTRACT)
    inputs = policy["inputs"]
    assert inputs["indexed_passages"] == 200
    assert inputs["query_rows_total"] == sum(
        inputs[name]
        for name in (
            "answerable_evaluation_queries",
            "unanswerable_calibration_queries",
            "unanswerable_evaluation_queries",
        )
    )
    gate = policy["acceptance"]
    assert gate["conjunctive"] is True
    assert gate["v7_invariants_required"] is True
    assert gate["minimum_treatment_recall_at_1"] == 0.8
    assert gate["minimum_treatment_recall_at_5"] == 0.95
    assert gate["paired_delta_lower_endpoint_strictly_greater"] is True
    assert gate["maximum_unanswerable_false_positive_rate"] == 0.05
    assert gate["maximum_shuffled_recall_at_1"] == 0.1


def test_v8_index_and_query_addressing_cannot_drift():
    policy = load(CONTRACT)
    treatment = policy["treatment"]
    encoder = treatment["address_encoder"]
    assert treatment["artifact_schema"] == "lecore.installed-memory-index.v2"
    assert treatment["manifest_digest_required"] is True
    assert treatment["legacy_last_position_sidecars_allowed"] is False
    assert encoder == {
        "name": "normalized_exponential_bundle_v1",
        "layer": "last",
        "decay": 0.99,
        "maximum_tokens": 200,
        "center_on_index_mean": True,
        "l2_epsilon": 1e-30,
        "similarity": "cosine",
        "index_and_query_encoder_must_match": True,
        "tie_break": "passage_id_ascending",
    }
    assert policy["controls"]["no_sidecar"] == {
        "required_behavior": "typed_refusal",
        "fallback_forbidden": True,
    }


def test_v8_execution_envelope_is_bounded_and_not_authorized():
    policy = load(CONTRACT)
    execution = policy["execution"]
    assert execution["ram_gib"] == 32
    assert execution["miniature_end_to_end_required_before_admission"] is True
    assert execution["timeout_seconds"] == 21600
    assert execution["maximum_compute_usd"] == 10.0
    assert execution["spot_instance_allowed"] is False
    assert policy["acceptance"]["maximum_peak_rss_mib"] == 28672


def test_source_audit_binds_main_pr31_and_immutable_v7_evidence():
    policy = load(CONTRACT)
    audit = load(AUDIT)
    assert audit["result"] == "current_main_already_ancestor_rebase_noop"
    assert audit["upstream"]["is_ancestor_of_verification_head"] is True
    assert audit["upstream"]["main_sha"] == policy["verification_base"][
        "upstream_main_sha"
    ]
    assert audit["verification"]["head_sha"] == policy["verification_base"][
        "verification_head_sha"
    ]
    assert audit["v7_source"]["patches_equivalent"] is True
    assert audit["v7_source"]["original_stable_patch_id"] == audit[
        "v7_source"
    ]["rebased_stable_patch_id"]
    assert audit["v7_evidence"]["evidence_sha256"] == policy[
        "prior_evidence"
    ]["evidence_sha256"]


def test_source_audit_matches_the_checked_out_verification_graph():
    audit = load(AUDIT)
    upstream = audit["upstream"]["main_sha"]
    verification = audit["verification"]["head_sha"]
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", upstream, verification],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0
    tree = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "%s:experiments/qwen35_acceptance/results/v7-cb3b1d2ac71c-accepted"
            % verification,
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    assert tree == audit["v7_evidence"]["result_tree_sha"]
    for name in ("v6", "v7"):
        pdf = ROOT / "output" / "pdf" / (
            "qwen35-%s-experiment-result.pdf" % name
        )
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        assert digest == audit["v7_evidence"]["%s_pdf_sha256" % name]


def test_v8_contract_digest_is_current():
    expected = (V8 / "contract.sha256").read_text(encoding="ascii").strip()
    actual = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert expected == actual
