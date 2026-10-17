Do Personal Agents Learn to Over-Personalize? Rubric Dropout for Memory-Grounded RL
Oct 16, 2026 · @Sean Noh
Summary
Rubric-based RL for personal agents can teach a model to bring up the user's memory where it doesn't belong: the personal-agent version of clickbait. This pilot trains a small model with GRPO against a cheap LLM judge and asks two things:

1. Does over-personalization appear? Measured on PersistBench, a published benchmark with its own judge from a different model family, validated against human rankings, and kept outside the training loop.
2. How much does a generic defense help? Rubric Dropout (Yang et al., 2026) is compared with no defense and with a hand-made rubric fix that knows exactly where the hole is.
   Headline question: Personalization rubrics invite memory over-use. A targeted fix stops it. How much does a generic defense recover without knowing the problem?
   Scope: a 1–2 day pilot, presented as an exploration.
   Why this question
   • Fixing the rubric only works when you know the hole. Real rubric datasets are generated automatically, with thousands of prompts and many criteria each, and nobody audits every criterion. Rubric Dropout is meant as a defense that doesn't need to know where the exploit is. This project measures how much that buys you compared with a targeted fix.
   • Personalization is especially exposed. Generic criteria like "Tailors the response to the user" are easiest to satisfy by mentioning a remembered fact, relevant or not. A TCP answer that opens with "Since you're training for a half marathon..." may earn credit from a cheap judge. For a real assistant that is a product failure: irrelevant at best, intrusive at worst.
   • The hack is measured outside the training loop. The Rubric Dropout paper inferred hacking from disagreement between its training judge and a stronger judge. Here, over-personalization is measured on PersistBench: a published benchmark built to provoke memory misuse, scored by its own judge (Kimi-K2-Thinking) from a different model family than the training judge, and validated against human rankings.
   Hypothesis
   Over-personalization appears in the base run. A generic personalization criterion rewards memory mentions on every prompt, so the shortcut is reinforced everywhere at once.
   Why dropout might help. Following the paper's analysis (Appendix A): before group standardization, dropout only scales the expected advantage by a constant (1 − f), which standardization removes. So dropout does not change which answers are favored on average; it adds noise. That noise is largest for answers whose advantage hinges on a single high-weight criterion, and smallest for answers that are broadly better than their group. An answer that wins only through the personalization criterion wins only on steps where that criterion is kept. An answer that wins on substance wins almost every step, because several substance criteria nearly always survive. The bet is that steady signals get reinforced more reliably than intermittent ones. The paper treats this as intuition, and so does this project.
   Prediction. Dropout slows the hack but doesn't prevent it. A memory mention costs almost nothing on the other criteria, so the push toward it never reverses; it just arrives less often. Dropout should land between Base and Fixed, probably closer to Base. In the paper, effects were modest (1–3 gold points on the medical pair, up to 7 on science at 8B) and appeared as slower decline after the peak, not a higher peak.
   Design
   Arms
   Arm
   Training rubric
   Represents
   Base
   Realistic rubrics (see below)
   No defense
   Dropout
   Same rubrics, 50% Rubric Dropout
   Generic defense
   Fixed
   Personalization criteria removed on memory-irrelevant queries
   Targeted fix (oracle upper bound; uses the relevance flag)
   Same code, data, hyperparameters and number of training steps across arms. Only the rubric file or the --dropout flag differs.
   Finding personalization criteria (Fixed arm). If rubrics are natural, personalization-like criteria may be worded differently across prompts. Classify every criterion with an LLM ("Does this criterion reward personalizing to the user?"), then hand-check about 50 classifications before using the labels. The same labels drive the rubric check below.
   Data
   About 500 training prompts, all LLM-generated, split roughly 50/50 between memory-relevant and memory-irrelevant queries.
   Each example has:
   • A memory profile of 8–10 facts (diet, location, job, hobbies, family, health constraints), as a list of strings like PersistBench's.
   • A query, either memory-relevant ("What should I make for dinner tonight?") or memory-irrelevant ("How does TCP handle packet loss?").
   • A rubric of about 10 weighted criteria (weights 1–5).
   • A relevance flag, used only by the Fixed arm and for analysis.
   Prompt format. Training prompts inject memories using PersistBench's system prompt template (or a very close match). If training and evaluation present memory differently, leakage changes could reflect the format mismatch rather than the training method.
   As built (Oct 17, datagen/): PersistBench-style data. 200 profiles x 5 queries = ~1,000 prompts, 80/20 train/held-out by profile. Memories: 4-16 "User ..." sentences across 3-5 of PersistBench's 9 domains (sensitive ones included). Queries: first-person, 2-4 sentences; relevant ones (50%) need >= 2 memories, irrelevant ones are cross-domain (query domain outside the profile's memory domains, ~30% adjacent). Diversity: code picks persona, domains, task type, and a situation drawn without replacement from a hand-skimmed situation bank (9 domains x 120). Gate: an LLM lists the memories that would materially change a good answer (empty for irrelevant, >= 2 for relevant); failures are rewritten up to 3 times, then dropped. Rubric + personalization labels stored per query; the injected criterion is appended at training time. No similarity pass: the generator never sees PersistBench content. Rubric writer stays gpt-4o-mini.
   Generation (original plan):
3. Generate about 100 profiles, with seeded attributes (occupation, age, life domains) for diversity.
4. Pair each profile with about 5 queries. For relevant queries, at least two profile facts must genuinely matter. For irrelevant queries, check the pairing against the profile (TCP is memory-relevant for a network engineer). Include some loosely adjacent irrelevant queries, since thematic bridges are where leakage is hardest to avoid.
5. Have an LLM write each rubric the normal way, with no hand-injected criteria.
6. Check that no generated item resembles a PersistBench item.
7. Split by profile, so no profile appears in more than one split.
   Day 1 checks (before training)
   Three quick checks, each of which could change the plan.
8. Step-0 PersistBench. Run the leakage and beneficial subsets on the untrained model. If it already leaks heavily, there is no room for leakage to rise, and the framing shifts to "does training reduce or worsen existing leakage, and does dropout change that?"
9. Rubric check: natural or injected. Count how often rubrics for memory-irrelevant queries include a personalization criterion.
   • Common: use the natural rubrics as they are. The hack arises on its own, which is the stronger result.
   • Rare: append "Personalizes the response to the user" (weight 3) to every rubric, and cite the measured rate as the justification. State clearly that this makes it a controlled test with a stronger dose.
   Report the rate either way. It is a small finding in its own right.
   Result (49 generated examples, gpt-4o-mini rubric writer and labeler): 61% of rubrics for memory-irrelevant queries include a personalization criterion, averaging 7% of rubric weight. Labels were hand-spot-checked after tightening the labeler's definition (a looser definition gave 90%, mostly from generic "engagement" criteria).
   Decision: inject "Personalizes the response to the user" (weight 3) into every training rubric, on top of the natural criteria. Justification: it appears naturally in most rubrics, but the natural payoff (see judge check) is too weak to rely on within a 1–2 day run. Frame as a controlled test with a stronger dose of a criterion that occurs naturally in 61% of rubrics.
10. Judge check: does the training judge pay for name-dropping? For about 10 irrelevant queries, write one answer with a gratuitous memory reference and one without, otherwise identical. Score both with the training judge. If the name-dropping version does not score higher, the hack cannot emerge regardless of the rubrics, and the setup needs rethinking before any training.
    Result (49 examples, 3 judge repeats): on natural rubrics, name-dropping adds +0.034 reward (helps 24, hurts 6, no change 19). With the injected criterion, +0.090 (helps 37, hurts 3, no change 9); the injected criterion passes 0% of clean answers and 69% of name-drop answers. Name-dropping also failed "concise" and "no irrelevant information" criteria 10 times, so the hack is not free, which favors dropout. Some unrelated criteria flipped despite identical content (halo or judge noise).
    Rerun on the PersistBench-style training set (Oct 17; 383 irrelevant queries, 1 judge repeat): natural rubrics +0.046 (helps 158, hurts 23, no change 202); injected +0.094 (helps 293, hurts 11, no change 79); injected criterion passes 3% of clean and 71% of name-drop answers. 87% of irrelevant rubrics carry a natural memory-use criterion. Name-dropping is nearly free here (215 criteria up vs 33 down), which by the reasoning above makes dropout less likely to help: prediction stays 'slows, does not stop'.
    Pending: a control with a neutral filler sentence of similar length, to separate the memory effect from "an added sentence" and judge noise.
    Training
    Setting
    Value
    Policy
    Qwen3-1.7B with LoRA, on one L40 (48GB). Runs are sequential, ~2.5–5 h each: Base first (doubles as the scouting run), then Dropout, then Fixed. 4B does not fit the timeline on one GPU
    Algorithm
    GRPO via TRL GRPOTrainer, vLLM for generation, LoRA if GPU < 80GB
    Rollouts per prompt
    8
    Learning rate
    About 1e-5 with LoRA, 1e-6 for a full fine-tune
    Reward judge
    One cheap API model from a different family than Kimi (e.g. gpt-4o-mini)
    Steps
    Set from the Base run's early checkpoints; long enough to see a leakage trend. The paper's gold peak was around step 240
    Reward. The judge grades all criteria in one call and returns a 0/1 verdict per criterion. The reward is the weighted fraction satisfied over kept criteria:
    R(x, y; m) = \frac{\sum_k m_k w_k s_k(x, y)}{\sum_k m_k w_k}
    Here s_k is the verdict, w_k the weight, and m_k is 1 if criterion k is kept. For Base and Fixed, every m_k is 1.
    Dropout.
11. Seed the mask with a hash of (prompt id, global step). All rollouts of a prompt in one step share the mask; the mask changes every step. GRPO compares rollouts within a group, so they must be scored on the same sub-rubric.
12. Drop round(0.5 × K) criteria uniformly at random, always keeping at least three.
13. Compute the reward on kept criteria only. No extra judge calls are needed, since the judge already returns every verdict.
14. Also log the full-rubric reward.
    Evaluation
    What
    When
    Purpose
    PersistBench subset, untrained model
    Day 1, before training
    Baseline; if leakage is already near the ceiling, rethink the framing
    PersistBench cross-domain leakage (~100 fixed samples)
    3–4 checkpoints per run
    Main metric: over-personalization
    PersistBench beneficial subset (~100 fixed samples)
    Same checkpoints
    Memory is still used where it helps; lower leakage must not come from ignoring memory
    Full-rubric training reward
    Logged every step
    Dropout isn't just learning less; plotted next to leakage, it shows the hacking gap
    PersistBench settings. Use one generation per sample at intermediate checkpoints. The default protocol generates three and counts a sample as failed if any one fails, which costs three times as much. Use the same samples at every checkpoint and the same memory system prompt as training. The judge is Kimi-K2-Thinking via OpenRouter; check the cost per checkpoint before committing to a schedule. The authors recommend the Inspect-native implementation over their own repository.
    Implementation decisions (Oct 17): (1) System prompt: PersistBench's generic_prompt.txt (from the paper's prompt ablation), used for both training and evaluation, rendered exactly as PersistBench renders memories. The default evaluation.txt is a ~20KB ChatGPT-style prompt with tool specs (~5k tokens); using it in training would multiply step time. Headline numbers are therefore not directly comparable to the paper's default-prompt results. (2) PersistBench files (generic prompt, the two judge prompts, the eval subset) are vendored unmodified in third_party/persistbench/ (commit d2ded18), and evaluate.py reproduces PersistBench's generation and judge call (Kimi-K2-Thinking on OpenRouter, temperature 0, Google Vertex routing, high reasoning effort) and failure-rate definitions. No second repo, server, or separate environment. (3) Fixed subset: 100 cross-domain items (random.Random(0).sample) + all 100 beneficial items. (4) PersistBench has no license file: keep this repo private, or ask the authors, before publishing. Beneficial items are reported two ways: PersistBench's code counts only rating 1 as a failure; the paper's text also counts partial use (rating 2). (5) vLLM pinned to 0.19.1: later releases need CUDA 13 (driver >= 580); eniac's driver is 570.
    Comparison protocol. Compare arms on the average over the last third of training, not only the final checkpoint. Hacking grows with training length, so final-step comparisons mix the effect with timing. All three arms train for the same number of steps.
    Noise. With one seed and about 100 samples, leakage rates carry roughly ±10 points of sampling noise near 50%. Describe where Dropout lands between Base and Fixed in rough terms ("closer to base," "about halfway") rather than as a precise percentage, and call small differences suggestive.
    Write-up:
15. Plot: PersistBench leakage over training steps, all three arms, with the training reward alongside.
16. Table: late-window leakage and beneficial scores per arm.
17. Two or three example answers to an irrelevant query, from late Base versus Dropout and Fixed.
18. The natural personalization-criterion rate from the rubric check, and the judge-check result.
    Possible outcomes
    • Leakage rises in Base, Dropout lands between Base and Fixed: the generic defense gives partial protection. Report roughly how close to Fixed it gets.
    • Dropout ≈ Base: dropout doesn't help against a nearly free, single-criterion hack. That is a concrete limitation of the method.
    • Dropout ≈ Fixed: the generic defense works as well as knowing the problem. A strong, surprising result; check that the beneficial score didn't drop.
    • No leakage rise in Base: the hack didn't emerge at this scale or run length. Report it honestly.
    Implementation
    Training runs as scripts (overnight, must survive disconnects). Use a notebook only for analysis.
    agent-personalization/
    ├── config.py # models, paths, settings
    ├── datagen/ # (a) data generation: profiles.py, queries.py, rubrics.py (+ injected criterion, labeler), check_examples.py
    ├── training/ # (b) judge.py (rubric grading, memory-mention check), reward.py (reward + dropout mask), train.py (next)
    ├── evaluation/ # (c) persistbench.py (Kimi judge, scoring), evaluate.py (run on a checkpoint)
    ├── utils/ # (d) llm.py (clients, retries, JSON), memory.py (prompt formatting), policy.py (vLLM), cache.py
    ├── third_party/persistbench/ # (e) vendored PersistBench files
    ├── checks/ # (f) one file per Day 1 check: judge, rubric, env, policy, persist
    ├── data/ # data only (JSON/JSONL)
    ├── outputs/ # checks/<dataset>/, checks/policy/, persistbench/<label>/
    └── analysis/analysis.ipynb # plots and examples (later)
    Run each section from the repo root: uv run python -m datagen check-examples | uv run python -m checks {judge,rubrics,env,policy,persist} | uv run python -m evaluation --label step0
    Training as built (Oct 17, training/train.py): TRL 1.13.0 GRPOTrainer, vLLM colocate on one L40, LoRA r=16 on all linear layers, lr 1e-5 constant, 16 prompts x 8 rollouts per step, beta=0, temperature 1.0, max completion 1024, 300 steps, seed 0. Rubric modes: natural (as generated; Base and Dropout) and fixed (personalization-labeled criteria removed on irrelevant queries). --dropout 0.5 drops floor(n/2) criteria per prompt per step (keep >= 3), seeded by (prompt id, step). Positive control: --personalization-weight 3 for ~50 steps. Injected criterion is a fallback only if natural rubrics are too weak in the pilot. Live metrics per step: reward, full-rubric reward, personalization-criterion pass rate on irrelevant answers, memory-mention rate on 16 sampled irrelevant answers (+ 10-step average). Outputs go to the required --output-dir: config.json, log.jsonl, rollouts/ every 10 steps, LoRA checkpoints every 25 steps.
      uv run python -m training.train --output-dir DIR/control --personalization-weight 3 --max-steps 50
      uv run python -m training.train --output-dir DIR/base
      uv run python -m training.train --output-dir DIR/dropout --dropout 0.5
      uv run python -m training.train --output-dir DIR/fixed --rubric fixed
      uv run python -m evaluation --label base-step200 --model DIR/base/checkpoints/checkpoint-200
    Log every checkpoint's scores and raw responses to JSONL (or Weights & Biases).
    Correctness checks
    Places where a silent bug would invalidate the result. Verify these by hand.
    [ ] One dropout mask per prompt group per step: same across rollouts, different across steps.
    [ ] Reward is the weighted fraction over kept criteria only, with at least three kept.
    [ ] Fixed arm removes personalization criteria only on irrelevant queries.
    [ ] Training prompts use the same memory system prompt as PersistBench evaluation.
    [ ] Evaluation never uses a dropped rubric.
    [ ] No PersistBench item, or close paraphrase, appears in training data.
    [ ] A 10-step smoke test runs end to end before any real run.
    Timeline
19. Day 1, morning: generate data and criterion labels; run the three Day 1 checks (step-0 PersistBench, rubric rate, judge check); write the judge prompt.
20. Day 1, afternoon: wire the judge into the reward, run the smoke test, launch all three arms (in parallel if GPUs allow).
21. Overnight: runs. Check the Base arm's early checkpoints to confirm leakage is moving.
22. Day 2: finish runs, evaluate checkpoints, plots, examples, README.
    If compute is tight, train all three arms for the same number of steps and evaluate Fixed at fewer checkpoints. Don't shorten Fixed: comparisons use the late window, and a shorter run could look good only because it never reached the hacking phase. Judge API costs are usually tens of dollars per run with a mini model; check before long runs.
    Limitations
    • One seed per arm, a small model, and synthetic training data: a pilot, not a conclusion.
    • If the criterion is injected, the hack is planted deliberately. It is realistic, since templated criteria do appear in rubric datasets, but it is a stronger dose than nature provides.
    • The Fixed arm uses the relevance flag, which a real pipeline wouldn't have. It is an upper bound, not a practical baseline.
    • PersistBench's judge is also an LLM, not ground truth, though it was validated against human rankings.
    • Runs may be too short to pass the peak, in which case all arms may look alike.
    • The method belongs to the Rubric Dropout paper. The contribution is the domain, the comparison against a targeted fix, and measuring the hack on an external benchmark.
    Next steps if it works
    • Short versus long completion caps, to test whether dropout helps more when memory tangents cost space (the "hack has a cost" hypothesis).
    • Block Dropout over semantically similar personalization criteria.
    • More seeds and a 4B model.
    Background
    Rubric Dropout (Yang et al., Scale AI, arXiv 2608.11669, Oct 2026). Qwen3 models (1.7B–8B) trained with GRPO against rubric rewards from a cheap judge, evaluated by a stronger gold judge. Proxy and gold scores rise together, then diverge as the policy learns template shortcuts that satisfy generic criteria. Dropping a random 30–50% of criteria each step (one mask per prompt group, full rubric at evaluation) gave up to +3 gold points on HealthBench-Hard and +7 on ResearchQA, mostly as slower decline after the peak. Reweighting criteria did worse than no intervention. Training rubrics averaged about 27–30 criteria. Mostly single runs.
    PersistBench (Pulipaka et al., ICML 2026). 500 human-validated memory–query pairs: 200 cross-domain leakage, 200 memory-induced sycophancy, 100 beneficial memory use. Across 18 models, median leakage failure was 53%. The authors' own mitigations are inference-time (defensive prompting, memory retrieval), and they suggest post-training objectives as a direction, which this project tests.
    Bing Explore Further (Shah et al., KDD 2025). Bing's recommender was trained mainly on clicks and drifted toward clickbait. The team added a validated GPT-4 quality signal alongside clicks, blending the two scores (quality weight 0.32) and accepting a 1.2% CTR drop for a large quality gain. In a blind study, humans sided with the LLM label 54% of the time and with clicks 6%. This project asks the follow-up question: once an LLM judge becomes the training signal, what is the new clickbait? For personal agents, the answer may be gratuitous memory use.
