"""Publish the realism / eval-awareness NLA artifacts as a public HF dataset.

    /home/debian/nanoNLA-multi-input/.venv-cpu/bin/python upload_realism_dataset.py
"""
import os
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd

REPO = "syvb/nla-qwen36-27b-realism"
HERE = Path(__file__).resolve().parent
TOKEN = open("/home/debian/.hf_token").read().strip()

# local file -> path inside the dataset repo
FILES = {
    # recursive "make-it-more-realistic" ladders of the blackmail eval
    "realism_v0.json": "realism_ladder/seed_v0.json",
    "realism_versions.json": "realism_ladder/glm5.2_ladder_v0-v20.json",
    "realism_versions_opus.json": "realism_ladder/opus4.8_ladder_v0-v20.json",
    # 100 system-prompt variations + eval-y panel scores + NLA verbalizations
    "realism_variant_seed.txt": "variants/seed_first3paragraphs.txt",
    "realism_variants.json": "variants/variants_raw.json",
    "realism_variants_scored.json": "variants/variants_scored.json",
    "realism_variants_verb_mat_rl.json": "variants/verbalizations_matryoshka_rl.json",
    "realism_variants_verb_std.json": "variants/verbalizations_standard.json",
    "realism_variants_line_scores.json": "variants/line_evalawareness_scores.json",
    "realism_variants_correlate_mat_rl.json": "variants/correlation_matryoshka_rl.json",
    "realism_variants_correlate_std.json": "variants/correlation_standard.json",
    "realism_variants_corr_mat_rl.png": "variants/correlation_matryoshka_rl.png",
    "realism_variants_corr_std.png": "variants/correlation_standard.png",
    # earlier: verbalize v0 + v1 with all 3 NLAs, where-does-it-flag analysis
    "realism_verb_realism_v0_v0_mat_rl.json": "verbalization_v0_v1/verb_v0_matryoshka_rl.json",
    "realism_verb_realism_v0_v0_mat_warm.json": "verbalization_v0_v1/verb_v0_matryoshka_warm.json",
    "realism_verb_realism_v0_v0_std.json": "verbalization_v0_v1/verb_v0_standard.json",
    "realism_verb_realism_versions_v1_mat_rl.json": "verbalization_v0_v1/verb_glm_v1_matryoshka_rl.json",
    "realism_verb_realism_versions_v1_mat_warm.json": "verbalization_v0_v1/verb_glm_v1_matryoshka_warm.json",
    "realism_verb_realism_versions_v1_std.json": "verbalization_v0_v1/verb_glm_v1_standard.json",
    "realism_verb_realism_versions_opus_v1_mat_rl.json": "verbalization_v0_v1/verb_opus_v1_matryoshka_rl.json",
    "realism_verb_realism_versions_opus_v1_mat_warm.json": "verbalization_v0_v1/verb_opus_v1_matryoshka_warm.json",
    "realism_verb_realism_versions_opus_v1_std.json": "verbalization_v0_v1/verb_opus_v1_standard.json",
    "realism_analyze.json": "verbalization_v0_v1/where_flag_stats.json",
    "realism_where.png": "verbalization_v0_v1/where_flag.png",
    "DATASET_README.md": "README.md",
}


def main():
    api = HfApi(token=TOKEN)
    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    ops = []
    for local, remote in FILES.items():
        p = HERE / local
        if p.exists():
            ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(p)))
        else:
            print(f"  SKIP missing {local}")
    api.create_commit(REPO, repo_type="dataset", operations=ops,
                      commit_message="realism / eval-awareness NLA artifacts (Qwen3.6-27B)")
    print(f"[uploaded] {len(ops)} files -> https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
