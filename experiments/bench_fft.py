#!/usr/bin/env python3
"""Benchmark: matmul vs FFT for compute_ldos at realistic scale."""
import numpy as np, time, os, sys

sys.path.insert(0, "/n/netscratch/heller_lab/Lab/yicheng/2D-Dirac-Propagation")

print(f"NumPy: {np.__version__}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"CPU count: {os.cpu_count()}")

def ldos_matmul(snaps, times, Es, chi):
    Ns=len(times); ny,nx=snaps.shape[2],snaps.shape[3]; N_sp=ny*nx; n_E=len(Es)
    phases = np.exp(1j*np.outer(Es, times))
    sf = snaps.reshape(Ns,2,N_sp); cf = chi.ravel()
    sA = sf[:,0,:]*cf[None,:]; sB = sf[:,1,:]*cf[None,:]
    return (np.abs(phases@sA)**2+np.abs(phases@sB)**2).reshape(n_E,ny,nx)

def ldos_fft(snaps, times, Es, chi, N_fft):
    Ns=len(times); dt=times[1]-times[0]
    ny,nx=snaps.shape[2],snaps.shape[3]; N_sp=ny*nx; n_E=len(Es)
    sf = snaps.reshape(Ns,2,N_sp); cf = chi.ravel()
    sA = sf[:,0,:]*cf[None,:]; sB = sf[:,1,:]*cf[None,:]
    mod = np.exp(1j*Es[0]*times)[:,None]
    A_pad = np.zeros((N_fft,N_sp),dtype=complex); B_pad = np.zeros((N_fft,N_sp),dtype=complex)
    A_pad[:Ns] = sA*mod; B_pad[:Ns] = sB*mod
    ftA = np.fft.ifft(A_pad,axis=0)[:n_E]*N_fft
    ftB = np.fft.ifft(B_pad,axis=0)[:n_E]*N_fft
    return (np.abs(ftA)**2+np.abs(ftB)**2).reshape(n_E,ny,nx)

# Small correctness test
np.random.seed(42)
Ns=201; dt_snap=0.5; ny=nx=32
snap_times = np.arange(Ns)*dt_snap
omegas = [2.5-0.3j, 3.2-0.5j, 4.1-0.2j]
modes = [np.random.randn(2,ny,nx)+1j*np.random.randn(2,ny,nx) for _ in range(3)]
snaps_s = np.zeros((Ns,2,ny,nx), dtype=complex)
for n in range(3):
    for k in range(Ns):
        snaps_s[k] += modes[n]*np.exp(-1j*omegas[n]*snap_times[k])
chi_s = np.ones((ny,nx))
N_fft_s = 500; dE_s = 2*np.pi/(N_fft_s*dt_snap)
E_s = 0.5+np.arange(200)*dE_s

r1=ldos_matmul(snaps_s,snap_times,E_s,chi_s)
r2=ldos_fft(snaps_s,snap_times,E_s,chi_s,N_fft_s)
print(f"\nSmall correctness: rel_diff={np.max(np.abs(r1-r2))/(np.max(np.abs(r1))+1e-30):.2e}")

# Realistic benchmarks
for label, Ns_r, n_E_r, ny_r, dt_r in [
    ("LDOS-shapes", 201, 300, 256, 0.5),
    ("Scar",        1001, 500, 256, 0.1),
]:
    nx_r=ny_r; N_sp_r=ny_r*nx_r
    times_r = np.arange(Ns_r)*dt_r
    dE_r = 0.025
    N_fft_r = max(int(np.round(2*np.pi/(dE_r*dt_r))), Ns_r)
    E_r = 0.5+np.arange(n_E_r)*dE_r
    snaps_r = np.random.randn(Ns_r,2,ny_r,nx_r)
    chi_r = np.ones((ny_r,nx_r))

    print(f"\n=== {label}: Ns={Ns_r}, n_E={n_E_r}, n={ny_r}, N_fft={N_fft_r} ===")

    t0=time.perf_counter(); _=ldos_matmul(snaps_r,times_r,E_r,chi_r); t_m=time.perf_counter()-t0
    print(f"  Matmul: {t_m:.2f}s")

    t0=time.perf_counter(); _=ldos_fft(snaps_r,times_r,E_r,chi_r,N_fft_r); t_f=time.perf_counter()-t0
    print(f"  FFT:    {t_f:.2f}s  (speedup: {t_m/t_f:.1f}x)")

    # Correctness on real-size random data
    r1_r=ldos_matmul(snaps_r,times_r,E_r[:50],chi_r)  # subset for speed
    r2_r=ldos_fft(snaps_r,times_r,E_r[:50],chi_r,N_fft_r)
    rel_r=np.max(np.abs(r1_r-r2_r))/(np.max(np.abs(r1_r))+1e-30)
    print(f"  Correctness (n_E=50 subset): rel_diff={rel_r:.2e}")

    mem_m = (n_E_r*Ns_r + Ns_r*N_sp_r*2 + n_E_r*N_sp_r*2)*16/1e9
    mem_f = (N_fft_r*N_sp_r*2 + n_E_r*N_sp_r*2)*16/1e9
    print(f"  Memory: matmul={mem_m:.1f}GB, FFT={mem_f:.1f}GB")
