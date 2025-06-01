## Artifact for more results and source code and data

Multi-agent systems (MASs) have emerged as a promising paradigm for automated code generation, demonstrating impressive performance on established benchmarks by decomposing complex coding tasks across specialized agents with different roles. Despite their prosperous development and adoption, their robustness remains pressingly under-explored, raising critical concerns for real-world deployment. 

This paper presents the first comprehensive study examining the robustness of MASs for code generation through a fuzzing-based testing approach. We design a fuzzing pipeline, incorporating semantic-preserving mutation operators and fitness functions, to effectively assess mainstream MASs across multiple datasets and LLMs. Our findings reveal substantial robustness flaws: MASs fail to solve 7.9\%--83.3\% of problems they initially resolved successfully after applying semantic-preserving mutations. Through comprehensive failure analysis, we identify a common yet largely overlooked source of robustness degradation: miscommunications between planning and coding agents, where plans lack sufficient detail and coding agents misinterpret intricate logic. To address this issue, we propose a repairing method, which encompasses multi-prompt generation and introduces a new monitor agent. Evaluation shows that our repairing effectively enhances the robustness of MASs by solving 40.0\%--76.9\% of identified failures.  Our work uncovers critical robustness flaws in MASs and provides effective mitigation strategies, contributing essential insights for developing more reliable MASs for code generation.

Our artifact include more experiment result for RQ2 and RQ3, prompt and examples for the mutation operators and monitor agent, and source code and experiment result of our paper.

### Prerequisites

For each MAS (Self-Collaboration, MetaGPT and PairCoder), you can find ` requirement.txt` in corresponding folders.

run` pip install -r requirement.txt `  to prepare the environment for each MAS.

### More Experiment Result

The full result of RQ2 can be found in `./More_results/RQ2_all_result/RQ2_all_result.pdf`

The full result of RQ3 can be found in `./More_results/RQ3_all_result/RQ3_all_result.pdf`

### Prompt and examples for the mutation operators and monitor 

Prompt and examples for the mutation operators can be found in `./More_results/Mutation_opertors/operators.pptx`

Prompt and examples for the monitor agent can be found in `./More_results/Monitor/monitor.pptx`

### Source code and experiments

Source code and experiment result for Self-Collaboration Code Generation (SCCG) can be found in `./Experiments/Self-Collaboration`. You can and run the scripts under  `./sh`  to regenerate the result.

Source code and experiment result for MetaGPT can be found in `./Experiments/metagpt`. You can and run the scripts under  `./z_scripts`  to regenerate the result.

Source code and experiment result for PairCoder can be found in `./Experiments/PairCoder`.You can and run the scripts under  `./z_scripts`  to regenerate the result.



