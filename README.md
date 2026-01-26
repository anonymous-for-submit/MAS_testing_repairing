## Artifact: source code and data

Multi-agent systems (MASs) have emerged as a promising paradigm for automated
code generation, demonstrating impressive performance on established benchmarks
by decomposing complex coding tasks across specialized agents with different
roles. Despite their prosperous development and adoption, the fundamental
mechanisms underlying their robustness remain poorly understood, raising critical
concerns for real-world deployment.
This paper conducts a systematic empirical study to uncover the internal
robustness flaws of MASs using an automated mutation-based methodology. By
designing a testing pipeline incorporating semantic-preserving mutation
operators and a novel fitness function, we assess mainstream MASs
across multiple datasets and LLMs. Our findings reveal substantial
robustness flaws: semantically equivalent inputs cause drastic performance
drops, with MASs failing to solve 7.9\%--83.3\% of problems they
initially resolved successfully after applying the semantic-preserving
mutations.

Through comprehensive failure analysis, we discover a fundamental cause
underlying these robustness issues: the \textit{planner-coder gap}, which
accounts for 75.3\% of failures. This gap arises from information loss
in the multi-stage transformation process where planning agents decompose
requirements into underspecified plans, and coding agents subsequently
misinterpret intricate logic during code generation.
Based on this formulated information transformation process, we propose
a \textit{repairing method} that mitigates information loss through
multi-prompt generation and introduces a monitor agent to bridge the
planner-coder gap. Evaluation shows that our repairing method
effectively enhances the robustness of MASs by solving 40.0\%--88.9\% of identified
failures. Re-execution of the testing process on the repaired MASs shows that the number of found failures
decreases up to 85.7\%, demonstrating that repaired MASs exhibit superior robustness.
Our work uncovers critical robustness
flaws in MASs and provides effective mitigation strategies, contributing
essential insights for developing more reliable MASs for code generation.

Our artifact include more experiment result for RQ2 and RQ3, prompt and examples for the mutation operators and monitor agent, and source code and experiment result of our paper.

## Prerequisites

For each MAS (Self-Collaboration, MetaGPT and PairCoder), you can find ` requirement.txt` in corresponding folders.

run` pip install -r requirement.txt `  to prepare the environment for each MAS.

## Source code and experiments

Source code and experiment result for Self-Collaboration Code Generation (SCCG) can be found in `./Experiments/Self-Collaboration`. You can and run the scripts under  `./sh`  to regenerate the result.

Source code and experiment result for MetaGPT can be found in `./Experiments/metagpt`. You can and run the scripts under  `./z_scripts`  to regenerate the result.

Source code and experiment result for PairCoder can be found in `./Experiments/PairCoder`.You can and run the scripts under  `./z_scripts`  to regenerate the result.

## More Experiment Result

The full result of RQ2 can be found in `./More_results/RQ2_all_result.pdf`

## Prompt and examples for the mutation operators and monitor 

Prompt and examples for the mutation operators can be found in `./More_results/Mutation_opertors/operators.pptx`

Prompt and examples for the monitor agent can be found in `./More_results/Monitor/monitor.pptx`

## Annotation result for MAS failure reasons

Manual annotation for MAS failure reasons are available in `./More_results/failure_categorization.xlsx`



