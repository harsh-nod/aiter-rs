"""CPU-only freeze checks and private fixture generation for OPUS full-K."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import statistics
from pathlib import Path

from tasks.opus_a16w16_persistent.adversarial_matrix import AITERSHA, validate_case

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HEADER = Path("csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh")
BASE_HEADER_SHA256 = "bf621cfe97d2b38a89ca668c57b32a10094aeef92b4fdbb723e530e0abd6946b"
SCHEMA = "aiter-rs-opus-persistent-fullk-matrix-v1"
TASK_SCHEMA = "aiter-rs-opus-persistent-header-task-v1"
SUPPORTED = (
    ("aligned-nooob", 8192, 4096, 256, 1300, "random"),
    ("m-tail-oob", 12287, 4096, 256, 300, "random"),
    ("n-tail-16-aligned", 16384, 2192, 512, 300, "random"),
    ("m-one-row-tail", 12033, 4096, 256, 300, "checkerboard"),
    ("n-first-vector-tail", 16384, 2064, 256, 300, "cancellation"),
    ("mn-combined-tail", 16383, 2064, 256, 300, "random"),
    ("k-min-even-loop", 8192, 4096, 128, 1300, "checkerboard"),
    ("xcd-padded-grid", 4096, 16384, 128, 1300, "checkerboard"),
)
SHAPES = {(m, n, k, kid) for _, m, n, k, kid, _ in SUPPORTED}
PUBLIC = {name: (m, n, k, kid, pattern) for name, m, n, k, kid, pattern in SUPPORTED}
PUBLIC_MATRIX_SHA256 = "840326059a47f5ec7f61006101f06b3061aa0ce2d1022e7805c75a2389adc77c"
WITHHELD_MATRIX_SHA256 = "84451f4660298138d813018b67f281ef0f99283b7fe50418a7358a5bcf5c795c"
PUBLIC_SEEDS = {
    "aligned-nooob": 95001, "m-tail-oob": 95002, "n-tail-16-aligned": 95003,
    "m-one-row-tail": 95011, "n-first-vector-tail": 95012,
    "mn-combined-tail": 95013, "k-min-even-loop": 95015, "xcd-padded-grid": 95016,
}
MATRIX_KEYS = {
    "schema", "aiter_sha", "arch", "gpu_sku", "layout", "input_dtype",
    "output_dtype", "split_k", "atol", "rtol", "cases",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_matrix(matrix: dict, *, withheld: bool) -> None:
    if set(matrix) != MATRIX_KEYS:
        raise ValueError("matrix fields changed")
    expected = {
        "schema": SCHEMA, "aiter_sha": AITERSHA, "arch": "gfx950",
        "gpu_sku": "AMD Instinct MI350X",
        "layout": "contiguous_[1,M,K]x[1,N,K]to[1,M,N]",
        "input_dtype": "bfloat16", "output_dtype": "bfloat16",
        "split_k": 0, "atol": 0.125, "rtol": 0.02,
    }
    if any(matrix.get(key) != value for key, value in expected.items()):
        raise ValueError("matrix contract changed")
    cases = matrix["cases"]
    expected_count = len(SUPPORTED) + 4 if withheld else len(SUPPORTED)
    if not isinstance(cases, list) or len(cases) != expected_count:
        raise ValueError(f"exactly {expected_count} full-K cases required")
    if len({case.get("id") for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    if not withheld and len({(case.get("m"), case.get("n"), case.get("k"), case.get("kid")) for case in cases}) != len(SHAPES):
        raise ValueError("exactly one public case per admitted shape/kid required")
    for case in cases:
        validate_case(case)
        shape = tuple(case[key] for key in ("m", "n", "k", "kid"))
        if shape not in SHAPES:
            raise ValueError("unadmitted shape or kid")
        if withheld:
            if case["pattern"] not in {"random", "checkerboard", "cancellation"} or case["source"] != "proposed":
                raise ValueError("withheld cases must be full-K random or adversarial BF16 inputs")
            if case["id"] in PUBLIC:
                raise ValueError("withheld case ID exposes a public fixture")
        else:
            name = case["id"]
            if name not in PUBLIC or (*shape, case["pattern"]) != PUBLIC[name]:
                raise ValueError("public shape/kid/pattern drifted")
            if case["seed"] != PUBLIC_SEEDS[name]:
                raise ValueError("public fixture seed drifted")
            if case["source"] != ("admitted_anchor" if name in {"aligned-nooob", "m-tail-oob", "n-tail-16-aligned"} else "proposed"):
                raise ValueError("public fixture provenance drifted")
    if not withheld and {case["id"] for case in cases} != set(PUBLIC):
        raise ValueError("public cases drifted")
    if withheld:
        random_cases = [case for case in cases if case["pattern"] == "random"]
        adversarial_cases = [case for case in cases if case["pattern"] != "random"]
        random_shapes = {
            tuple(case[key] for key in ("m", "n", "k", "kid"))
            for case in random_cases
        }
        seeds = [case["seed"] for case in random_cases]
        adversarial_rows = {
            (*tuple(case[key] for key in ("m", "n", "k", "kid")), case["pattern"])
            for case in adversarial_cases
        }
        expected_adversarial = {
            (m, n, k, kid, pattern)
            for _, m, n, k, kid, pattern in SUPPORTED if pattern != "random"
        }
        if (random_shapes != SHAPES or len(random_cases) != len(SHAPES)
                or len(set(seeds)) != len(seeds)
                or any(seed in PUBLIC_SEEDS.values() for seed in seeds)
                or len(adversarial_cases) != 4
                or adversarial_rows != expected_adversarial):
            raise ValueError("withheld matrix needs fresh random on every shape and adversarial full-K cases")


def validate_task(task: dict, public: dict) -> None:
    validate_matrix(public, withheld=False)
    if set(task) != {
        "schema", "task_id", "aiter_sha", "editable_path", "base_header_sha256",
        "entry", "public_matrix", "public_matrix_sha256", "withheld_matrix_sha256",
        "withheld_generator_sha256",
        "compiler_image_id", "compiler_hipcc_sha256", "host_gpu_report_sha256",
        "correctness", "performance", "task_mode", "scored_eligible",
    }:
        raise ValueError("task fields changed")
    if task.get("schema") != TASK_SCHEMA or task.get("aiter_sha") != AITERSHA:
        raise ValueError("task source pin changed")
    if task.get("task_id") != "gfx950.opus_a16w16_persistent.fullk.header.v1":
        raise ValueError("task identifier changed")
    if task.get("editable_path") != HEADER.as_posix() or task.get("base_header_sha256") != BASE_HEADER_SHA256:
        raise ValueError("editable source boundary changed")
    if task.get("entry") != "aiter.ops.opus.opus_bmm" or task.get("public_matrix") != "public_matrix.json":
        raise ValueError("operator boundary changed")
    if task.get("public_matrix_sha256") != PUBLIC_MATRIX_SHA256 or sha256(HERE / "public_matrix.json") != PUBLIC_MATRIX_SHA256:
        raise ValueError("public fixture raw hash changed")
    if task.get("scored_eligible") is not False or task.get("task_mode") != "no_feedback_candidate":
        raise ValueError("task cannot advertise scored eligibility or feedback")
    if task.get("compiler_image_id") != "sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b":
        raise ValueError("compiler image changed")
    if task.get("compiler_hipcc_sha256") != "9a0fa4bc274155e7add34dc1897bfc08f2f5d6732ad8b9102b7d7a86c0e3d781":
        raise ValueError("hipcc compiler changed")
    if task.get("host_gpu_report_sha256") != "4b7b8c1d2bf204b465422b93c182fdb8cf4aa478bb47ecf2494380b05d364d4c":
        raise ValueError("MI350X host attestation changed")
    perf = task.get("performance", {})
    if perf != {
        "boundary": "preallocated_opus_bmm_exact_kid_split_k_0",
        "method": "32_call_graph_replay_gpu_events_paired_alternating",
        "warmup_direct": 5, "warmup_graph": 2, "pairs": 20,
        "max_relative_mad": 0.05,
        "max_candidate_to_aiter_ratio_per_case": 1.05,
        "separate_improvement_target": "geometric_mean_candidate_to_aiter_ratio_le_0.95",
    }:
        raise ValueError("performance boundary or gates changed")
    correct = task.get("correctness", {})
    if correct != {
        "oracle": "torch_fp32_matmul_of_bf16_operands", "atol": 0.125, "rtol": 0.02,
        "steps": ["phase0_first", "phase0_repeat", "phase1_reuse", "phase1_alternate_output", "phase0_restore", "phase0_guard_perturbation"],
        "checks": ["every_element_finite_and_within_tolerance", "input_immutable", "input_and_output_canaries", "inactive_output_immutable", "phase0_reproduced", "exact_persistent_dispatch"],
    }:
        raise ValueError("independent oracle contract changed")
    if task.get("withheld_matrix_sha256") != WITHHELD_MATRIX_SHA256:
        raise ValueError("private matrix commitment changed")


def clean_jit_paths(source: Path, baseline_jit: Path, candidate_jit: Path, overlay: Path) -> None:
    paths = [source.resolve(), baseline_jit.resolve(), candidate_jit.resolve(), overlay.resolve()]
    if len(set(paths)) != 4:
        raise ValueError("source, overlay, and JIT paths must be disjoint")
    for path in paths[1:]:
        if path.is_relative_to(REPO) or path.is_relative_to(paths[0]):
            raise ValueError("overlay and JIT paths must be outside source/repo")
    if any(left.is_relative_to(right) or right.is_relative_to(left)
           for index, left in enumerate(paths[1:]) for right in paths[index + 2:]):
        raise ValueError("overlay and JIT paths must not contain one another")
    if any(path.exists() for path in paths[1:]):
        raise ValueError("overlay and JIT paths must be new for this control")


def private_matrix(public: dict) -> dict:
    validate_matrix(public, withheld=False)
    private = {key: value for key, value in public.items() if key != "cases"}
    random_cases = [
        {**case, "id": secrets.token_hex(12), "seed": secrets.randbits(63),
         "pattern": "random", "source": "proposed"}
        for case in public["cases"]
    ]
    adversarial = [
        {**case, "id": secrets.token_hex(12), "seed": secrets.randbits(63),
         "source": "proposed"}
        for case in public["cases"] if case["pattern"] in {"checkerboard", "cancellation"}
    ]
    private["cases"] = random_cases + adversarial
    validate_matrix(private, withheld=True)
    return private


def write_private(path: Path, matrix: dict) -> str:
    path = path.absolute()
    if path.resolve().is_relative_to(REPO):
        raise ValueError("private matrix must stay outside the repository")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.stat().st_mode & 0o077:
        raise ValueError("private matrix parent must be mode 700")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(matrix, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return sha256(path)


def performance_gate(rows: list[dict]) -> dict:
    if len(rows) != len(SUPPORTED) or {row.get("id") for row in rows} != set(PUBLIC):
        raise ValueError("one result per public bucket required")
    ratios = []
    for row in rows:
        ratio = row.get("candidate_to_aiter_ratio")
        mads = (row.get("aiter_relative_mad"), row.get("candidate_relative_mad"))
        if any(type(x) not in (float, int) or not math.isfinite(x) for x in (ratio, *mads)):
            raise ValueError("finite numeric graph evidence required")
        if ratio <= 0 or min(mads) < 0:
            raise ValueError("invalid ratio or MAD")
        ratios.append(ratio)
    qualified = all(
        row.get("correctness_pass") is True
        and row.get("post_graph_readonly_pass") is True
        and row.get("exact_dispatch") is True
        and row.get("same_boundary") is True
        and row.get("pid_gate") is True
        and max(row["aiter_relative_mad"], row["candidate_relative_mad"]) <= 0.05
        for row in rows
    )
    return {
        "evidence_qualified": qualified,
        "per_bucket_noninferior": qualified and all(ratio <= 1.05 for ratio in ratios),
        "geomean_ratio": statistics.geometric_mean(ratios),
        "separate_improvement_target_met": qualified and statistics.geometric_mean(ratios) <= 0.95,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-private", type=Path)
    parser.add_argument("--verify-private", type=Path)
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--aiter-source", type=Path)
    parser.add_argument("--baseline-jit", type=Path)
    parser.add_argument("--candidate-jit", type=Path)
    parser.add_argument("--overlay-tree", type=Path)
    args = parser.parse_args()
    public = json.loads((HERE / "public_matrix.json").read_text(encoding="utf-8"))
    task = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    validate_task(task, public)
    if sum(bool(value) for value in (args.create_private, args.verify_private, args.check_paths)) > 1:
        parser.error("private generation, verification, and runtime-path preflight are separate steps")
    if args.check_paths:
        if not all((args.aiter_source, args.baseline_jit, args.candidate_jit, args.overlay_tree)):
            parser.error("runtime-path preflight requires source, two JIT paths, and overlay")
        clean_jit_paths(args.aiter_source, args.baseline_jit, args.candidate_jit, args.overlay_tree)
        print(json.dumps({"clean_runtime_paths": True}))
    elif args.create_private:
        commitment = write_private(args.create_private, private_matrix(public))
        print(json.dumps({"withheld_matrix_sha256": commitment, "generator_sha256": sha256(Path(__file__))}))
    elif args.verify_private:
        path = args.verify_private.resolve()
        if path.is_relative_to(REPO) or sha256(path) != WITHHELD_MATRIX_SHA256:
            raise ValueError("private matrix location or commitment mismatch")
        validate_matrix(json.loads(path.read_text(encoding="utf-8")), withheld=True)
        print(json.dumps({"withheld_commitment_valid": True}))
    else:
        print(json.dumps({"task_valid": True, "public_matrix_sha256": sha256(HERE / "public_matrix.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
