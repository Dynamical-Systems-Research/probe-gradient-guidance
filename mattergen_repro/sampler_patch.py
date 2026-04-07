#!/usr/bin/env python3
"""Patch script: applies best-of-K support to pc_sampler.py on the spark."""

SAMPLER_PATH = "/home/jarrodbarnes/mattergen/mattergen/diffusion/sampling/pc_sampler.py"

with open(SAMPLER_PATH) as f:
    content = f.read()

# The block to replace: from candidate generation through proposal_scorer.choose
OLD_BLOCK = '''                    candidate_batch, candidate_mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )
                    if self._proposal_scorer is not None:
                        batch, mean_batch = self._proposal_scorer.choose(
                            current_batch=batch,
                            current_mean_batch=mean_batch,
                            proposed_batch=candidate_batch,
                            proposed_mean_batch=candidate_mean_batch,
                            t=t,
                            prepare_batch_fn=self._prepare_batch_for_self_correction,
                        )
                    else:
                        batch, mean_batch = candidate_batch, candidate_mean_batch'''

NEW_BLOCK = '''                    candidate_batch, candidate_mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )
                    if self._proposal_scorer is not None:
                        num_k = getattr(self._proposal_scorer, "num_proposals", 1)
                        if num_k > 1 and hasattr(self._proposal_scorer, "choose_best_of_k"):
                            # Best-of-K: first candidate already generated above
                            k_candidates = [(candidate_batch, candidate_mean_batch)]
                            for _k in range(num_k - 1):
                                k_samples_means = apply(
                                    fns=fns,
                                    broadcast={"t": t, "dt": dt},
                                    x=batch,
                                    score=score,
                                    batch_idx=self._multi_corruption._get_batch_indices(batch),
                                )
                                k_cand, k_mean = _mask_replace(
                                    samples_means=k_samples_means,
                                    batch=batch, mean_batch=mean_batch, mask=mask,
                                )
                                k_candidates.append((k_cand, k_mean))
                            batch, mean_batch = self._proposal_scorer.choose_best_of_k(
                                current_batch=batch,
                                current_mean_batch=mean_batch,
                                candidates=k_candidates,
                                t=t,
                                prepare_batch_fn=self._prepare_batch_for_self_correction,
                            )
                        else:
                            batch, mean_batch = self._proposal_scorer.choose(
                                current_batch=batch,
                                current_mean_batch=mean_batch,
                                proposed_batch=candidate_batch,
                                proposed_mean_batch=candidate_mean_batch,
                                t=t,
                                prepare_batch_fn=self._prepare_batch_for_self_correction,
                            )
                    else:
                        batch, mean_batch = candidate_batch, candidate_mean_batch'''

if OLD_BLOCK not in content:
    print("ERROR: Could not find the exact block to replace in pc_sampler.py")
    print("The file may have been modified. Manual patching required.")
    exit(1)

new_content = content.replace(OLD_BLOCK, NEW_BLOCK)

with open(SAMPLER_PATH, 'w') as f:
    f.write(new_content)

print("pc_sampler.py patched successfully with best-of-K support.")
