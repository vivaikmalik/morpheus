# archive/

Frozen records from the cleanup of feature/dep-aware-decoding. Files here describe state at archival time; they are not maintained. See paper_claim_index.md for the map from paper claims to source files.

## audits/

Verification reports comparing paper claims to code and to eval CSVs. Includes the consistency audit of main.tex against the eval data, the technical audit of architecture and training claims, and the source audit for the §5.4 ring-count number. Pre-cleanup investigation notes (REFACTOR_PRECHECK, TRAINING_RESUME_PLAN) also live here.

## investigations/

Analysis and audit reports from the outputs/ directory at the time the paper was written. Includes the dep-aware decoding investigation, the token capability probe results for the 20-epoch and 40-epoch checkpoints, the replication experiment results, and the bug, branch, ring, and tokenizer audits.

## old_code/

Code paths superseded by the canonical pipeline. Includes the v1 RL trainer (replaced by training/rl_reinforce_v2.py), the standalone evaluate.py, two earlier text-conditioning training scripts, the one-off EOS resume helper, and a duplicate prompt collator. inference/novelty_check.py, inference/diversity_check.py, and inference/distribution_analysis.py read data produced by the archived inference_evaluate_v1.py. Re-running the analysis pipeline requires either unarchiving this script or replacing it.
