##Imports##
import numpy as np
import re
import os
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from pathlib import Path


def compute_h_eff_from_blockmesh(total_N_list, block_length=1.0, blocks=2,
                                 yExpansion=10.09757454):
    h_eff_list = []
    dy_list_of_lists = []

    # geometric multiplier inside a block:
    R = float(yExpansion)

    for Ntot in total_N_list:
        N_block = Ntot // blocks   # cells per block

        # block physical lengths
        L1 = block_length
        L2 = block_length
        if N_block == 1:
            dy1 = np.array([L1])
        else:
            q1 = R**(1.0/(N_block-1)) if R != 1.0 else 1.0
            if abs(q1 - 1.0) < 1e-12:  # uniform
                a1 = L1 / N_block
                dy1 = np.full(N_block, a1)
            else:
                a1 = L1 * (q1 - 1.0) / (q1**N_block - 1.0)  
                dy1 = a1 * q1**np.arange(N_block)
        if N_block == 1:
            dy2 = np.array([L2])
        else:
            q2 = (1.0 / q1) if q1 != 0 else 1.0
            if abs(q2 - 1.0) < 1e-12:
                a2 = L2 / N_block
                dy2 = np.full(N_block, a2)
            else:
                a2 = L2 * (q2 - 1.0) / (q2**N_block - 1.0)
                dy2 = a2 * q2**np.arange(N_block)
        full_dy = np.concatenate([dy1, dy2])
        h_eff = full_dy.mean() 
        h_eff_list.append(h_eff)
        dy_list_of_lists.append(full_dy)

    return h_eff_list, dy_list_of_lists

N = [80, 40, 20]
h_eff, dy_arrays = compute_h_eff_from_blockmesh(N, block_length=1.0, blocks=2, yExpansion=10.09757454)
h = np.array(h_eff) 

expdatafile = ["profiles/chan180.means","profiles/chan180.reystress","balances/chan180.kbal"]

datafile = ["sampleEpsilon/100000/y_turbulencePropertiesepsilon.xy",
            "sampleG/100000/y_productionRate.xy",
            "sampleK/100000/y_turbulencePropertiesk.xy",
            "sampleR/100000/y_turbulencePropertiesR.xy",
            "sampleU/100000/y_U.xy",
            "wallShearStress1/0/wallShearStress.dat"]
datadict = {
                "20": 
                        {"e": 0,
                         "g": 0,
                         "k": 0,
                         "r": 0,
                         "u": 0,
                         "ss" : 0},
                "40": 
                        {"e": 0,
                         "g": 0,
                         "k": 0,
                         "r": 0,
                         "u": 0,
                         "ss" : 0},
                "80": 
                        {"e": 0,
                         "g": 0,
                         "k": 0,
                         "r": 0,
                         "u": 0,
                         "ss" : 0}}
for i in range(len(N)):
    for j,key in zip(range(len(datafile)),datadict[str(N[i])].keys()):
        parts = datafile[j].split("/", 2)
        direct = "/".join(parts[:2])
        
        os.chdir(f"{N[i]}/{direct}")
        if ".dat" in parts[-1]:
            with open(parts[-1]) as fh:
                text = fh.read()
            
            matches = re.findall(r'\(\s*([-0-9eE.+ ]+?)\s*\)', text)
            vecs = [m.split() for m in matches]
            datadict[str(N[i])][key] = pd.DataFrame(vecs, dtype=float,
                                                    columns=["tau_x", "tau_y", "tau_z"]) 
        else:
            datadict[str(N[i])][key] =  pd.read_csv(parts[-1],sep="\s+",header=None)
        print("Finished reading",parts[-1])
        os.chdir("C:/Users/Bshol_mkzpcxg/OneDrive - Texas A&M University/Grad 1 (Fall 25)/NUEN 628/Proj/data")
    
def load_mkm_file(filename):
    data_start = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped and (stripped[0].isdigit() or stripped[0] == "-"):
                data_start = i
                break

    df = pd.read_csv(
        filename,
        sep = '\s+',
        skiprows=data_start,
        header=None,
        comment="#",
    )

    return df

#Main GCI function 
def GCI(h, f, p_expected):
    def fun(pobs):
        return P_OBS(h, f, pobs)
    pobs = fsolve(fun, p_expected)[0]
    r = h[1] / h[0]

    if abs((pobs - p_expected) / p_expected) > 0.1:
        F = 3.0
        p = min(max(0.5, pobs), p_expected)
    else:
        F = 1.25
        p = p_expected

    GCI = F * abs(f[1] - f[0]) / (r**p - 1)
    return GCI, pobs

def P_OBS(h, f, p):
    r23 = h[2] / h[1]
    r12 = h[1] / h[0]

    return (f[2] - f[1]) / (r23**p - 1) - (r12**p * (f[1] - f[0])) / (r12**p - 1)

nu = 0.006147          

for n in datadict.keys():
    tau_w = datadict[n]["ss"]["tau_x"].astype(float).mean()
    datadict[n]["tau_w"] = tau_w  

for n in datadict.keys():

    Udf = datadict[n]["u"]
    y = Udf.iloc[:,0].values
    U = Udf.iloc[:,1].values
    uTau =  np.sqrt(abs(datadict[n]["tau_w"]))
    y_plus = y * uTau / nu
    U_plus = U / uTau
    ReTau = uTau/nu
    datadict[n]["u"] = pd.DataFrame({
        "y": y,
        "U": U,
        "y_plus": y_plus,
        "U_plus": U_plus
    })

    kdf = datadict[n]["k"]
    y = kdf.iloc[:,0].values
    kval = kdf.iloc[:,1].values
    k_plus = kval / (uTau**2)

    datadict[n]["k"] = pd.DataFrame({
        "y": y,
        "k": kval,
        "k_plus": k_plus
    })

    edf = datadict[n]["e"]
    y = edf.iloc[:,0].values
    eps = edf.iloc[:,1].values
    eps_plus = eps * nu / (abs(datadict[n]["tau_w"]))**1.5

    datadict[n]["e"] = pd.DataFrame({
        "y": y,
        "epsilon": eps,
        "epsilon_plus": eps_plus
    })

    gdf = datadict[n]["g"]
    y = gdf.iloc[:,0].values
    gval = gdf.iloc[:,1].values
    g_plus = gval * nu / (abs(datadict[n]["tau_w"]))**1.5

    datadict[n]["g"] = pd.DataFrame({
        "y": y,
        "G": gval,
        "G_plus": g_plus
    })

    rdf = datadict[n]["r"]
    y = rdf.iloc[:,0].values
    stresses = rdf.iloc[:,[1,2,4,6]].values
    stresses_plus = stresses / (uTau**2)

    datadict[n]["r"] = pd.DataFrame(
        np.column_stack([y, stresses, stresses_plus]),
        columns=["y","Ruu","Ruv","Rvv","Rww","Ruu+","Ruv+","Rvv+","Rww+"]
    )

    print(f"Normalized data for grid {n}")
print(f"\nuTau = {uTau}\n\nReTau = {ReTau}\n")

bench = {}

for fpath in expdatafile:
    name = Path(fpath).name 
    root, extension = os.path.splitext(fpath)
    fullpath = Path("chan180") / fpath
    print(fullpath)
    df = load_mkm_file(fullpath)
    bench[name] = df
    print(f"Loaded benchmark file: {name}, shape = {df.shape}")

bench_yplus = bench["chan180.means"].iloc[:, 1].values
bench_Uplus = bench["chan180.means"].iloc[:, 2].values
bench_eplus = abs(bench["chan180.kbal"].iloc[:, 2].values)
bench_kplus = 0.5*(bench["chan180.reystress"].iloc[:, 2].values + bench["chan180.reystress"].iloc[:, 3].values + bench["chan180.reystress"].iloc[:, 4].values)
bench_Ruu = bench["chan180.reystress"].iloc[:, 2].values
bench_Ruv = abs(bench["chan180.reystress"].iloc[:, 5].values)
bench_Rvv = bench["chan180.reystress"].iloc[:, 3].values
bench_Rww = bench["chan180.reystress"].iloc[:, 4].values

components = {
    "u+": {"df_key": "u", "col": "U_plus", "bench": bench_Uplus},
    "Ruu+": {"df_key": "r", "col": "Ruu+", "bench": bench_Ruu},
    "Ruv+": {"df_key": "r", "col": "Ruv+", "bench": bench_Ruv},
    "Rvv+": {"df_key": "r", "col": "Rvv+", "bench": bench_Rvv},
    "Rww+": {"df_key": "r", "col": "Rww+", "bench": bench_Rww},
    "e+": {"df_key": "e", "col": "epsilon_plus", "bench": bench_eplus},
    "k+": {"df_key": "k", "col": "k_plus", "bench": bench_kplus}
}

errors = {n: {} for n in datadict.keys()}

for n in datadict.keys():
    print(f"\n===== Computing errors for grid {n} =====")
    uTau = np.sqrt(abs(datadict[n]["tau_w"]))
    
    for comp_name, comp_info in components.items():
        df_key = comp_info["df_key"]
        col = comp_info["col"]
        bench_vals = comp_info["bench"]
    
        CFD = datadict[n][df_key]

        yplus = CFD["y"].values * uTau / nu
    
        val = CFD[col].values

        if comp_name == "Ruv+":
            val = np.abs(val)
    
        sort_idx = np.argsort(yplus)
        yplus_sorted = yplus[sort_idx]
        val_sorted = val[sort_idx]
    
        #f_interp = interp1d(yplus_sorted, val_sorted, bounds_error=False, fill_value="extrapolate")
        f_interp = CubicSpline(yplus_sorted, val_sorted)
        CFD_interp = f_interp(bench_yplus)

        
        point_err = CFD_interp - bench_vals
        L2_error = np.sqrt(np.mean(point_err**2))
        Linf_error = np.max(np.abs(point_err))
        errors[n][f"{comp_name}_interp"] = CFD_interp
        errors[n][f"{comp_name}_L2"] = L2_error
        errors[n][f"{comp_name}_Linf"] = Linf_error
        errors[n][f"{comp_name}_pointwise_error"] = point_err

        
        print(f"{comp_name} | L2 error: {L2_error:.4e}, Linf error: {Linf_error:.4e}")

def assemble_interp_list(comp):
    return list(zip(*[errors[str(Ni)][f"{comp}_interp"] for Ni in N]))  

for comp_name in components.keys():
    print(f"\n===== GCI for {comp_name} =====")
    interp_list = assemble_interp_list(comp_name)
    for j, vals in enumerate(interp_list):
        GCI_val, POBS_val = GCI(h, vals, 1)
        print(f"y+ index {j} | {comp_name} GCI: {GCI_val:.4e}, POBS: {POBS_val:.4f}")


plt.figure(figsize=(8,6))

for comp_name in components.keys():
    point_err = errors[str(N[0])][f"{comp_name}_pointwise_error"]
    
    plt.plot(bench_yplus, point_err, marker='o', linestyle='-', label=comp_name)

plt.xlabel("y+")
plt.ylabel("Error (OpenFOAM - Benchmark)")
plt.xscale("log")
plt.yscale("linear")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.title("Pointwise Error of OpenFOAM vs DNS Benchmark")
plt.tight_layout()
plt.show()

for comp_name, comp_info in components.items():
    plt.figure(figsize=(8,6))
    plt.plot(bench_yplus, comp_info["bench"], 'k-', linewidth=2, label="Benchmark")

    for n in N:
        interp_vals = errors[str(n)][f"{comp_name}_interp"]
        plt.plot(bench_yplus, interp_vals, marker='o', linestyle='--', label=f"OpenFOAM N={n}")
    
    plt.xlabel("y+")
    plt.ylabel(f"{comp_name}")
    plt.xscale("log")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title(f"{comp_name} | OpenFOAM vs DNS Benchmark")
    plt.tight_layout()
    plt.show()


N_fine = N[0]
n_key = str(N_fine)

for comp_name, comp_info in components.items():

    CFD_interp = errors[n_key][f"{comp_name}_interp"]
    bench_vals = comp_info["bench"]

    interp_list = assemble_interp_list(comp_name)

    GCI_vals = []
    for vals in interp_list:
        GCI_val, POBS_val = GCI(h, vals, 2)   # p_expected = 2
        GCI_vals.append(GCI_val)

    GCI_vals = np.array(GCI_vals)
    err_bar = 2 * GCI_vals

    plt.figure(figsize=(8,6))
    plt.plot(bench_yplus, bench_vals, 'k-', linewidth=2, label="Benchmark")
    plt.errorbar(
        bench_yplus, CFD_interp,
        yerr=err_bar,
        fmt='o', markersize=4, capsize=3, elinewidth=1,
        label=f"OpenFOAM N={N_fine} (±2·GCI)"
    )

    plt.xscale("log")
    plt.xlabel("y+")
    plt.ylabel(comp_name)
    plt.title(f"{comp_name} vs DNS – Error Bars")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


n_key = str(N_fine)

print("\n\n=======================================")
print("DISCRETIZATION ERROR VS COMPARISON ERROR")
print("============================================\n")

for comp_name, comp_info in components.items():
    print(f"\n===== {comp_name} =====")

    CFD_interp = errors[n_key][f"{comp_name}_interp"]
    bench_vals = comp_info["bench"]

    # Compute comparison error
    comp_error = CFD_interp - bench_vals

    # Compute discretization error = 2*GCI(y+) at each y+ point
    interp_list = assemble_interp_list(comp_name)
    GCI_vals = []
    for vals in interp_list:
        GCI_val, POBS_val = GCI(h, vals, 2)
        GCI_vals.append(GCI_val)

    GCI_vals = np.array(GCI_vals)
    disc_error = 2 * GCI_vals

    # Print a table-like structure
    print(f"{'y+':>10s} {'CompError':>15s} {'2*GCI (DiscErr)':>18s}")

    for yp, ce, de in zip(bench_yplus, comp_error, disc_error):
        print(f"{yp:10.4f} {ce:15.6e} {de:18.6e}")

    print("\nSummary for component:", comp_name)
    print(f"  Mean |CompError|  = {np.mean(np.abs(comp_error)):.4e}")
    print(f"  Mean 2*GCI        = {np.mean(disc_error):.4e}")
    print(f"  Max |CompError|   = {np.max(np.abs(comp_error)):.4e}")
    print(f"  Max 2*GCI         = {np.max(disc_error):.4e}")
