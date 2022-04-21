# SDE Project Code

## Overview
The project leverages the model proposed by the paper to conduct a deeper comparitive 
analysis on how students(novice developers) and proffesionals(experienced developers).
  
## Running the code
1. Install dependencies
   ```python
   pip install matplotlib
   pip install numpy
   pip install python-bidi
   ```

2. Copy the data(csv files) on whom you want to perform the analysis from `experienced_developers_results`
   or `novice_developers_results` and place them in the src folder before continuing to below steps.
   
3. Run `src/1_split_data.py` to split the data into manual and automatic processable questions. It generates 
   two files `results-other.csv` and `results-preprocessed.csv`. 

4. Run `src/2_manual_process.py` to view the data from `results-other.csv` that needs to be processed manually. 
   Process the data and then move to next step.

5. Run `src/3_analyse_results.py` to analyse the data and output the stats and graphs in the result folder.
