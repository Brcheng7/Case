#This is the run only code that only prints the deliverables. 

import os
import pandas as pd
from pathlib import Path
import numpy as np

os.chdir(os.path.expanduser("~/Downloads")) # Change to Downloads folder
print("Current working directory:", os.getcwd())

Case_name = "95k_km_case" # Change for other cases, i.e. 206k_km_case or 300k_km_case
root_dir = os.path.expanduser(f"~/Downloads/{Case_name}") #Folder should be located at Downloads Folder
file_info = []

for foldername, subfolders, filenames in os.walk(root_dir):
    for filename in filenames:
        full_path = os.path.join(foldername, filename)
        rel_path = os.path.relpath(full_path, root_dir)
        rel_path_posix = Path(rel_path).as_posix() 
        parent_folder = os.path.basename(foldername)
        file_info.append({
            'Path': rel_path_posix,
            'Name': filename,
            'ParentFolder': parent_folder
        })

df_files = pd.DataFrame(file_info) # Al files are stored with their relative path

# HPPC tests were selected for parameter estimation since cyclic data may not have rest periods after current pulses
hppc_csvs = df_files[df_files['Name'].str.contains('HPPC.csv', na=False)].reset_index(drop = True) # This dataframe contains all HPPC.csv containing test file paths (similar could be done for .mat files)

# SOC dependent parameter fitting requires HPPC tests first Beginning of life (BOL) condition is used since usable capacity will be equal to nominal capacity  
index_hppc = hppc_csvs[hppc_csvs['ParentFolder']== "Cycle 0"].index[0] # This represents the cycle used for anlalysis below BOL conditions are used as a start 
filepath = hppc_csvs.iloc[index_hppc,0]
file_cycle = hppc_csvs.iloc[index_hppc,2]

def find_header_line(filepath, header_name='Time Stamp'):
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.strip().startswith(header_name):
                return i
    return None  # if header not found

header_line = find_header_line(os.path.join(root_dir, filepath), header_name='Time Stamp') #find where the test data start in the selected csv 

if header_line is not None:
    df_read = pd.read_csv( os.path.join(root_dir, filepath), skiprows=header_line) # Read CSV 
else:
    raise ValueError("Time Stamp line not found in file")

df = df_read.iloc[1:].reset_index(drop=True) #first row includes units etc, remove them
df = df.loc[:,'Time Stamp':'Capacity']

df['Time Stamp'] = pd.to_datetime(df['Time Stamp'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Current'] = pd.to_numeric(df['Current'], errors='coerce')
df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
df['Prog Time'] = pd.to_timedelta(df['Prog Time'], errors='coerce')
df['Step Time'] = pd.to_timedelta(df['Step Time'], errors='coerce')
df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
df['dt']= df["Prog Time"].diff().dt.total_seconds()
df.loc[0,'dt'] = 0

current_threshold = 0.02  # 0.02 Amps considering noise in the measurement
voltage_change_threshold = 0.002  # 2 mV difference between 2 consecutive times

window_points = 150 # 15 second window at 100 ms sampling rate
half_window = window_points // 2


def is_voltage_stable_rolling(idx):
    if idx < half_window or idx + half_window >= len(df):
        return False
    window = df.iloc[idx - half_window : idx + half_window + 1]['Voltage']
    return (window.max() - window.min()) < voltage_change_threshold

rest_mask = (df['Current'].abs() < current_threshold) & df.index.to_series().apply(is_voltage_stable_rolling)

df['OCV_estimate'] = np.nan  

# Get indices where rest condition is satisfied
rest_indices = df.index[rest_mask]

# For each rest index, take the mean of the last 5 stable voltage values
for idx in rest_indices:
    if idx >= 4:  # Ensure at least 5 previous points are available
        ocv_value = df.loc[idx - 4:idx, 'Voltage'].mean()
        df.at[idx, 'OCV_estimate'] = ocv_value

Q_nom = 2.5  #  nominal capacity (Ah)
Soc = (1- abs(df.Capacity.values)/Q_nom)*100
SoC = pd.DataFrame(Soc)
df['SoC'] = SoC

from scipy.interpolate import PchipInterpolator

df_ocv = df[~np.isnan(df['OCV_estimate'])].copy() #filter out only present OCV values
df_ocv['SoC_rounded'] = df_ocv['SoC'].round(2)  # round to 2 decimal places

ocv_lookup = df_ocv.groupby('SoC_rounded')['OCV_estimate'].mean().reset_index() # Group by SoC and calculate mean of OCV_estimate
ocv_lookup.loc[-1,'SoC_rounded'] = 100
ocv_lookup.loc[-1,'OCV_estimate'] = df.loc[0,'Voltage']

ocv_to_soc = PchipInterpolator(ocv_lookup.OCV_estimate, ocv_lookup.SoC_rounded, extrapolate=True) # for mapping in the future
soc_to_ocv = PchipInterpolator(ocv_lookup.SoC_rounded, ocv_lookup.OCV_estimate, extrapolate=True) # for mapping in the future

df_step = df.copy()
df_step['is_one'] = df_step['Step'] == 1
# Find start of each run of 1s or non-1s
df_step['change'] = df_step['Step'].ne(df_step['Step'].shift())
# Get the indices where these runs start
start_indices = df_step.index[df_step['change'] & df_step['is_one']].tolist()
start_indices.append(len(df_step)) 
# Slice between these indices
list_of_dfs = [df.iloc[start_indices[i]:start_indices[i+1]] for i in range(len(start_indices)-1)]

from scipy.optimize import curve_fit

def double_exp(t, A1, tau1, A2, tau2, V_inf):
        return V_inf + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) # The eqn for Voltage relaxation part (after voltage drop occurs)
parameters_pulses = []
for seq in range(len(list_of_dfs)):
    pulse_df = list_of_dfs[seq].reset_index(drop = True)
    pulse_threshold = 0.85  # amps, noise floor

    # Identify pulse start and end indices based on current changes
    is_pulse = np.abs(pulse_df["Current"]) > pulse_threshold
    pulse_changes = np.diff(is_pulse.astype(int))

    # pulse_changes will be 1 where a pulse starts and -1 where it ends
    raw_starts = np.where(pulse_changes == 1)[0] + 1
    raw_ends = np.where(pulse_changes == -1)[0] + 1

    # Filter based on current stability
    valid_starts = []
    valid_ends = []
    for start, end in zip(raw_starts, raw_ends):
        pulse_currents = pulse_df.loc[start+1:start+50, 'Current'] 
        if pulse_currents.max() - pulse_currents.min() < pulse_threshold:  # Stability threshold the start and end current difference should be less than 1 A for pulses
            valid_starts.append(start)
            valid_ends.append(end)
        else:
            print("")

    pulse_starts = np.array(valid_starts)
    pulse_ends = np.array(valid_ends)

    time = range(len(pulse_df))
    
    for pul in range(len(pulse_starts)):
        pulse_start = pulse_starts[pul]-5 # get few time steps before for Voltage reading before pulse
        pulse_end = pulse_ends[pul]
        time_try = 200
        if np.any(np.abs(pulse_df.Current[pulse_end:pulse_end+time_try]) > 0.1):
            t_exp = 100
            
        else:   
            t_exp = 200
            
        pulse_final = pulse_end + t_exp 
        segment = pulse_df[pulse_start:pulse_final].reset_index(drop=True)

        delta = segment['Current'].diff().abs()
        sudden_changes = segment.index[delta > 0.5]
        
        V_start = segment.Voltage[sudden_changes[0]-1]
        V_drop = segment.Voltage[sudden_changes[0]+1]
        I = segment.Current[sudden_changes[0]] # Current changes from 0 to pulse current
        R0 = abs(V_start - V_drop)/abs(I) #R0 is dV/dI, since pulses have both increase and decrease, use absolute values

        end_current = abs(segment.Current[sudden_changes[1]])<0.1 #for relaxation curve fitting current must be ~ 0. Consider noise with 0.05 A threshold
        
        if end_current:
            relax_starts = sudden_changes[1]
            relax_ends = pulse_final - pulse_start
            time_relaxation = relax_ends - relax_starts
            t_relax = np.linspace(0, time_relaxation-1, time_relaxation) # numpy array format
            v_relaxation = segment.Voltage[relax_starts:relax_ends].reset_index(drop=True)
            v_relax = v_relaxation.values # numpy array format
            i_relaxation = segment.Current[relax_starts:relax_ends].reset_index(drop=True)
            
            assert np.all(np.abs(i_relaxation) < 0.1), "Current is not zero in relaxation segment!"

                
            p0 = [0.02, 5, 0.01, 50, v_relax[-1]]
            bounds = ([-1, 0.01, -1, 0.01, v_relax.min()], [1, 500, 1, 1000, v_relax.max()])

        # Fit
            params_ext, _ = curve_fit(double_exp, t_relax, v_relax, p0=p0, bounds=bounds, maxfev=10000)
            A1, tau1, A2, tau2, V_inf = params_ext

            pulse_current = (segment[segment.index< sudden_changes[1]]['Current'].iloc[-1])

            # Calculate R and C
            R1 = A1 / pulse_current
            C1 = tau1 / R1 if R1 != 0 else np.nan
            R2 = A2 / pulse_current
            C2 = tau2 / R2 if R2 != 0 else np.nan

            results = [seq, pul, pulse_final, v_relax[-1], pulse_current, R0, R1, R2, C1, C2]
            parameters_pulses.append(results)
            
        else:
            print("Not 0 or stable Current")   
    
Map = pd.DataFrame(parameters_pulses, columns=["segment", "pulse", "OCV_time", "OCV","pulse_current", "R0", "R1", "R2", "C1", "C2"])
Map["SOC"] = Map["OCV"].apply(ocv_to_soc)
Map['Type'] = Map['pulse_current'].apply(lambda x: 'charge' if x > 0 else ('discharge' if x < 0 else 'rest'))
Map.to_excel("ECM_parameters_raw.xlsx", index=False) 

Map_ch = Map[Map.Type == "charge"].copy()
Map_dch = Map[Map.Type == "discharge"].copy()

def moving_average(data, w):
    return pd.Series(data).rolling(window=w, center=True, min_periods=1).mean().values
Map_ch["R0_ma"] = moving_average(Map_ch["R0"], 3)
Map_ch["R1_ma"] = moving_average(Map_ch["R1"], 3)
Map_ch["R2_ma"] = moving_average(Map_ch["R2"], 3)
Map_ch["C1_ma"] = moving_average(Map_ch["C1"], 5)
Map_ch["C2_ma"] = moving_average(Map_ch["C2"], 5)

Map_dch["R0_ma"] = moving_average(Map_dch["R0"], 3)
Map_dch["R1_ma"] = moving_average(Map_dch["R1"], 3)
Map_dch["R2_ma"] = moving_average(Map_dch["R2"], 3)
Map_dch["C1_ma"] = moving_average(Map_dch["C1"], 5)
Map_dch["C2_ma"] = moving_average(Map_dch["C2"], 5)

Map_ch.to_excel("ECM_parameters_charge_smoothed.xlsx", index=False) 
Map_dch.to_excel("ECM_parameters_discharge_smoothed.xlsx", index=False) 

print("ECM parameters for charge and discharge segments are saved to excel files.")
print("Charge Table Snippet:",Map_ch.head())
print("Discharge Table Snippet:",Map_dch.head())

from scipy.interpolate import interp1d

interp_charge_R0 = interp1d(Map_ch[Map_ch['Type'] == 'charge']['SOC'],
                           Map_ch[Map_ch['Type'] == 'charge']['R0_ma'],
                           kind='linear', fill_value='extrapolate')

interp_discharge_R0 = interp1d(Map_dch[Map_dch['Type'] == 'discharge']['SOC'],
                              Map_dch[Map_dch['Type'] == 'discharge']['R0_ma'],
                              kind='linear', fill_value='extrapolate')

interp_charge_R1 = interp1d(Map_ch[Map_ch['Type'] == 'charge']['SOC'],
                              Map_ch[Map_ch['Type'] == 'charge']['R1_ma'],
                              kind='linear', fill_value='extrapolate')

interp_discharge_R1 = interp1d(Map_dch[Map_dch['Type'] == 'discharge']['SOC'],
                              Map_dch[Map_dch['Type'] == 'discharge']['R1_ma'],
                              kind='linear', fill_value='extrapolate')

interp_charge_R2 = interp1d(Map_ch[Map_ch['Type'] == 'charge']['SOC'],
                              Map_ch[Map_ch['Type'] == 'charge']['R2_ma'],
                              kind='linear', fill_value='extrapolate')

interp_discharge_R2 = interp1d(Map_dch[Map_dch['Type'] == 'discharge']['SOC'],
                              Map_dch[Map_dch['Type'] == 'discharge']['R2_ma'],
                              kind='linear', fill_value='extrapolate')

interp_charge_C1 = interp1d(Map_ch[Map_ch['Type'] == 'charge']['SOC'],
                              Map_ch[Map_ch['Type'] == 'charge']['C1_ma'],
                              kind='linear', fill_value='extrapolate')

interp_discharge_C1 = interp1d(Map_dch[Map_dch['Type'] == 'discharge']['SOC'],
                              Map_dch[Map_dch['Type'] == 'discharge']['C1_ma'],
                              kind='linear', fill_value='extrapolate')

interp_charge_C2 = interp1d(Map_ch[Map_ch['Type'] == 'charge']['SOC'],
                              Map_ch[Map_ch['Type'] == 'charge']['C2_ma'],
                              kind='linear', fill_value='extrapolate')

interp_discharge_C2 = interp1d(Map_dch[Map_dch['Type'] == 'discharge']['SOC'],
                              Map_dch[Map_dch['Type'] == 'discharge']['C2_ma'],
                              kind='linear', fill_value='extrapolate')

def simulate_ecm(test_profile):

    Time = pd.to_timedelta(test_profile["Step Time"]) 
    dt = Time.diff().dt.total_seconds()[1]    # time step in seconds get from the test step time difference
    soc_init = ocv_to_soc(test_profile.Voltage[0])

    n = len(test_profile)
    soc = soc_init
    V_rc1= 0.0
    V_rc2 = 0.0

    soc_history = np.zeros(n)
    voltage_history = np.zeros(n)

    for t in range(n):
        
        I = test_profile.Current[t]

        # Clamp SOC to [0, 100]
        soc = max(0, min(100, soc))

        # Select parameters by current sign and SOC
        if I > 0:
            R0 = interp_charge_R0(soc)
            R1 = interp_charge_R1(soc)
            C1 = interp_charge_C1(soc)
            R2 = interp_charge_R2(soc)
            C2 = interp_charge_C2(soc)
        elif I < 0:
            R0 = interp_discharge_R0(soc)
            R1 = interp_discharge_R1(soc)
            C1 = interp_discharge_C1(soc)
            R2 = interp_discharge_R2(soc)
            C2 = interp_discharge_C2(soc)
        else:
            # If no current, keep previous params
            R0 = interp_discharge_R0(soc)
            R1 = interp_discharge_R1(soc)
            C1 = interp_discharge_C1(soc)
            R2 = interp_discharge_R2(soc)
            C2 = interp_discharge_C2(soc)

        # Update RC voltages (discrete-time response)
        exp1 = np.exp(-dt / (R1 * C1))
        exp2 = np.exp(-dt / (R2 * C2))

        V_rc1_t = V_rc1 * exp1 + R1 * (1 - exp1) * (-I) # due to convention of charging results in negative current
        V_rc2_t = V_rc2 * exp2 + R2 * (1 - exp2) * (-I)

        # Calculate OCV at current SOC
        OCV = soc_to_ocv(soc)

        # Terminal voltage from ECM
        Vt = OCV - (-I * R0) - V_rc1_t- V_rc2_t

        V_rc1 = V_rc1_t
        V_rc2 = V_rc2_t

        # Update SOC based on current and time step
        Q_nom = 2.5  
        soc -= abs(I * dt/3600) / (Q_nom) * 100  
        
        soc_history[t] = soc
        voltage_history[t] = Vt

    return soc_history, voltage_history

test_profile = df
segment_test1 = test_profile[0:400].reset_index(drop=True)

test_profile = pd.read_csv("Dis.csv")
segment_test2 = test_profile[0:200].reset_index(drop=True)
test_profile = pd.read_csv("WLTP95.csv")
segment_test3 = test_profile[0:200].reset_index(drop=True)


soc_sim1, voltage_sim1 = simulate_ecm(segment_test1)
soc_sim2, voltage_sim2 = simulate_ecm(segment_test2)
soc_sim3, voltage_sim3 = simulate_ecm(segment_test3)

def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted)**2))

rmse1 = calculate_rmse(segment_test1.Voltage, voltage_sim1)
mae1 = calculate_mae(segment_test1.Voltage, voltage_sim1)

rmse2 = calculate_rmse(segment_test2.Voltage, voltage_sim2)
mae2 = calculate_mae(segment_test2.Voltage, voltage_sim2)

rmse3 = calculate_rmse(segment_test3.Voltage, voltage_sim3)
mae3 = calculate_mae(segment_test3.Voltage, voltage_sim3)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(voltage_sim1, label='Simulated Voltage')
plt.plot(segment_test1.Voltage, label='Measured Voltage', alpha=0.7)
plt.ylabel('Voltage (V)')
plt.xlabel('Time Steps')
plt.legend()
ax2 = plt.gca().twinx()
ax2.plot(segment_test1.Current, label='Current', color='red', linestyle = '--', alpha=0.5)
plt.ylabel('Current (A)')
plt.ylim(-3,1)
plt.title(f'ECM Simulatation  for Pulse Segment \n RMSE: {rmse1:.3f}, MAE: {mae1:.3f}')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(voltage_sim2, label='Simulated Voltage')
plt.plot(segment_test2.Voltage, label='Measured Voltage', alpha=0.7)
plt.ylabel('Voltage (V)')
plt.xlabel('Time Steps')
plt.legend()
ax2 = plt.gca().twinx()
ax2.plot(segment_test2.Current, label='Current', color='red', linestyle = '--', alpha=0.5)
plt.ylabel('Current (A)')
plt.ylim(-3,1)
plt.title(f'ECM Simulatation for Discharge Segment \n RMSE: {rmse2:.3f}, MAE: {mae2:.3f}')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(voltage_sim3, label='Simulated Voltage')
plt.plot(segment_test3.Voltage, label='Measured Voltage', alpha=0.7)
plt.ylabel('Voltage (V)')
plt.xlabel('Time Steps')
plt.legend()
ax2 = plt.gca().twinx()
ax2.plot(segment_test3.Current, label='Current', color='red', linestyle = '--',alpha=0.5)
plt.xlabel('Time Steps')
plt.ylabel('Current (A)')
plt.title(f'ECM Simulatationfor WLTP segment \n RMSE: {rmse3:.3f}, MAE: {mae3:.3f}')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

