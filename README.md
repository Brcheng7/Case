# Case

This repository contains the analysis of a battery model, including parameter fitting, validation against various test cycles (pulse, discharge, and WLTP), and a discussion of the results. The original work was presented as a Jupyter Notebook, a .py file to get required results and this README summarizes its key aspects. The notebook html verison with results is also provided. The verisons of the libraries used are found in requirements.txt.

The core methodology revolves around an Equivalent Circuit Model (ECM) to simulate battery behavior. The steps involved are:

Data Preparation:

Loading and preprocessing experimental data from battery tests (e.g., current, voltage, temperature, time). Reading HPPC, Discharge and WLTP csv files in the Cycle 0 folder in the 95k case study.

OCV and SoC extraction:
OCV is found by finding the rest voltage of the battery after pulses. SoC was calculated both by coulomb counting and by dividing the capacity measured by the nominal capacity.
Then OCV to SoC mapping was done using cubic interpolation.

Parameter Fitting:

Curvefit of scipy is used to fit relaxation data to the second-order ECM model. From OCV values, SoC mapping was performed to find SoC-dependent Parameter fitting. Moving average was used to smooth the parameters wrt SoC.

Model Validation:

The fitted model is validated against different driving cycles or test conditions not used during the fitting process. Additionally, the used pulse data was also simulated.

Validation is performed on:

Pulse Test: To assess dynamic response.
Discharge Test: To check long-term behavior and capacity estimation.
WLTP Cycle: To evaluate performance under a realistic driving profile.

The voltage error (difference between simulated and actual voltage) is a key metric for validation.

## Key takeaways:
- The timesteps were not consistent in the test data, so segmentation was required
- Some pulses were not used due to not having a constant current or not enough magnitude
- Charge and discharge parameters were separated since they showed different trends
- A moving average was applied to smooth out parameter change wrt SoC
- The validation was not satisfactory, but the trends for WLTP cycle were captured, which show promise since the real case scenario -car driving- is very dynamic.
- R0 parameter estimation probably has the biggest error since the voltage drop during a pulse is not very well captured.


## Instructions for the code

First, the 95 km case should be in the Downloads folder and it should be renamed as 95k_km_case. The working directory of the scripts is the Downloads folder, it reads profile data for Discharge and WLTP cycle from Downloads, and the hppc data is read from 95k_km_case folder. 

The libraries used and their versions are given in [requirements.txt](requirements.txt),

[run_wo_plots.py](run_wo_plots.py) file runs on the terminal provided that the above conditions are met. It prints out parameter mapping and validation plots. 

[Case_Study_case95k_wResults.html](Case_Study_case95k_wResults.html) is the report of the code that was created as a Jupyter notebook.

[Case_Study_case95k_wresults.ipynb](Case_Study_case95k_wresults.ipynb) is the Jupyter notebook with results.

[Case_Study_case95k_clean.ipynb](Case_Study_case95k_clean.ipynb) is the clean version without the results.








