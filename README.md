# Sirismt


This repository contains the code used in our SiriSMT project. The project consists of three main parts: the simulation, refinement phase, and the integration phase, which results in a complete solving strategy.

## Prerequisites

Before running the code, make sure to install the following dependencies:

1. **Z3 Solver**  
   - Required version: `Z3 version 4.12.x`
   - Installation instructions: [Z3 GitHub Repository](https://github.com/Z3Prover/z3)
   
2. **KLEE Symbolic Execution Engine**  
   - Required version: `KLEE version 2.x`
   - Installation instructions: [KLEE Installation Guide](https://klee.github.io/getting-started/)

3. **Python Dependencies**  
   Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt

## Project Structure

1. **Graph Building Phase**

    In this phase, we transform the existing SMT formula dataset into a graph structure. This step is critical as it encodes the formulas into a format suitable for subsequent processing.
    
    Due to certain reasons, using a **relative path** when passing \[dataset_folder\] may cause errors. You can try passing an **absolute path** to resolve the issue. 

    **How to run:**
    
    ```shell
    cd experiments
    ./build_graphs.sh -j [pararrel_num] -d [dataset_folder]
    ```

2. **Simulation and Refinement Phases**

    The simulation phase involves training the reinforcement learning model and producing candidate tactic sequences. During this phase, the model iteratively interacts with the refinement phase, where the sequences are pruned, and new ones are generated. This feedback loop allows us to refine the solving strategy.

    More customization options can be achieved by modifying the example YAML file in experiments/configs.

    **How to run:**
    
    ```shell
    python3 -u trainer.py --train_data [your_folder] --seed 0 --device cuda:0 --without_integration
    ```

3. **Integration Phase**

    The final phase integrates the refined tactic sequences to construct a complete solving strategy for the SMT solver. The learned strategies from the simulation and refinement are combined to achieve an optimized solution.

    **How to run:**
    
    ```shell
    python3 -u sirismt/combiner/combiner.py --valid_data [your_folder] \
                                            --tuned_strategies [tuned_strategies_file] \
                                            --cache_path [your_path] \
                                            --batch_size [thread_num] \
                                            --out_file [output_file]
    ```

## Reproducing Evaluation

To reproduce the evaluation of our method, you can run the following commands:

**How to run:**

```shell
python3 -u validator.py --main_strategy [your_strategy] --test_data [your_folder] --timeout [timeout] --batch_size [thread_num]
```

---

Feel free to reach out if you have any questions!


