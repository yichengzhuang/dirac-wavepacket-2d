#!/usr/bin/env python3
"""
Exact diagonalisation of 2D Dirac Hamiltonian on a finite grid with
infinite-mass boundary condition (mass confinement in an elliptical region).

Tests the even/odd density mirror theorem (Theorem 6.1):
  For each Kramers doublet, the Mx-even and Mx-odd states satisfy
  ρ_even(x, -y) = ρ_odd(x, y), and |<φ_odd | My φ_even>| = 1.

Hamiltonian:
  H = v_F (σ_x p_x + σ_y p_y) + m(r) σ_z
where m(r) = 0 inside the ellipse, M (large) outside.
"""

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ── Parameters ──
n   = 32        # grid size (matrix is 2*n² × 2*n²)
L   = 6.0       # domain [-L/2, L/2)²
vF  = 1.0       # Fermi velocity
# Scalar potential (preserves T = σ_y K)
k_harm = 3.0    # harmonic strength
a   = 1.5       # ellipse semi-axis x
b   = 0.9       # ellipse semi-axis y


# ── Symmetry operations ──

def _fft_flip(arr, axis):
    """FFT-consistent reflection: index j → (n-j) % n along given axis."""
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)

def apply_Mx(phi):
    """M_x = σ_y P_x:  (φ_A, φ_B) → (-i φ_B(-x,y), i φ_A(-x,y))"""
    up, dn = phi[0], phi[1]
    return np.stack([-1j * _fft_flip(dn, axis=1),
                      1j * _fft_flip(up, axis=1)], axis=0)

def apply_My(phi):
    """M_y = σ_x P_y:  (φ_A, φ_B) → (φ_B(x,-y), φ_A(x,-y))"""
    up, dn = phi[0], phi[1]
    return np.stack([_fft_flip(dn, axis=0),
                     _fft_flip(up, axis=0)], axis=0)

def density(phi):
    return np.abs(phi[0])**2 + np.abs(phi[1])**2

def overlap(a_phi, b_phi, dx):
    return (np.vdot(a_phi[0].ravel(), b_phi[0].ravel()) +
            np.vdot(a_phi[1].ravel(), b_phi[1].ravel())) * dx**2

def normalize(phi, dx):
    norm2 = np.sum(density(phi)) * dx**2
    return phi / np.sqrt(norm2) if norm2 > 0 else phi


# ── Grid ──
dx_val = L / n
x = np.linspace(-L / 2, L / 2, n, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="xy")

# k-space grid (FFT frequencies)
dk = 2 * np.pi / L
kx = np.fft.fftfreq(n, d=dx_val) * 2 * np.pi
ky = np.fft.fftfreq(n, d=dx_val) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing="xy")

# Zero out Nyquist frequencies to preserve T = σ_y K symmetry.
# On even grids, the Nyquist mode k = n/2 maps to itself under k→-k,
# breaking the antiunitary symmetry T. Zeroing it restores [T, H] = 0.
KX_safe = KX.copy()
KY_safe = KY.copy()
ny = n // 2
KX_safe[ny, :] = 0; KX_safe[:, ny] = 0
KY_safe[ny, :] = 0; KY_safe[:, ny] = 0

# ── Scalar potential: harmonic ellipse (preserves T = σ_y K) ──
V_field = 0.5 * k_harm * (X**2 / a**2 + Y**2 / b**2)


# ── Build Hamiltonian matrix ──
N = 2 * n * n  # total dimension

def apply_H(psi_flat):
    """Apply H to a flattened spinor vector of length 2*n*n."""
    psi = psi_flat.reshape(2, n, n)

    # Kinetic: v_F (σ_x p_x + σ_y p_y)
    # σ_x p_x + σ_y p_y has off-diagonal: (kx - i ky) upper-right, (kx + i ky) lower-left
    psi_A_k = np.fft.fftn(psi[0])
    psi_B_k = np.fft.fftn(psi[1])

    # Zero Nyquist modes to preserve T = σ_y K symmetry
    # (Nyquist frequency has no -k partner on even grids)
    H_psi_A_k = vF * (KX_safe - 1j * KY_safe) * psi_B_k
    H_psi_B_k = vF * (KX_safe + 1j * KY_safe) * psi_A_k

    result = np.zeros_like(psi, dtype=complex)
    result[0] = np.fft.ifftn(H_psi_A_k)
    result[1] = np.fft.ifftn(H_psi_B_k)

    # Scalar potential: V·I (preserves T)
    result[0] += V_field * psi[0]
    result[1] += V_field * psi[1]

    return result.ravel()


print(f"Building {N}×{N} Hamiltonian matrix (n={n}, L={L}) ...")
print(f"Harmonic ellipse: a={a}, b={b}, k_harm={k_harm}")

H_matrix = np.zeros((N, N), dtype=complex)
for col in range(N):
    if col % 1000 == 0:
        print(f"  column {col}/{N} ...")
    e = np.zeros(N, dtype=complex)
    e[col] = 1.0
    H_matrix[:, col] = apply_H(e)

# Verify Hermiticity
herm_err = np.linalg.norm(H_matrix - H_matrix.conj().T) / np.linalg.norm(H_matrix)
print(f"Hermiticity error: {herm_err:.2e}")
assert herm_err < 1e-10, f"H is not Hermitian! error = {herm_err}"

# ── Diagonalise ──
print("Diagonalising ...")
eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
print(f"Done. Eigenvalue range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")

# ── Select bound states: |E| well below M ──
E_cut = 15.0    # select states with |E| below this
bound_mask = np.abs(eigenvalues) < E_cut
bound_idx = np.where(bound_mask)[0]
print(f"Bound states (|E| < {E_cut}): {len(bound_idx)}")

if len(bound_idx) == 0:
    print("No bound states found. Try increasing n or decreasing M.")
    exit(1)

# ── Group into Kramers doublets ──
# Eigenvalues come in degenerate pairs (Kramers).
# Group by proximity.
bound_E = eigenvalues[bound_idx]
doublets = []
used = set()
for i in range(len(bound_idx)):
    if i in used:
        continue
    # find nearest unmatched partner
    best_j = None
    best_dE = np.inf
    for j in range(i + 1, len(bound_idx)):
        if j in used:
            continue
        dE = abs(bound_E[j] - bound_E[i])
        if dE < best_dE:
            best_dE = dE
            best_j = j
        if dE > 0.5:  # energies are sorted, can break early
            break
    if best_j is not None and best_dE < 0.5:
        doublets.append((bound_idx[i], bound_idx[best_j]))
        used.add(i)
        used.add(best_j)
    else:
        # unpaired (shouldn't happen for true Kramers)
        doublets.append((bound_idx[i], None))
        used.add(i)

print(f"Kramers doublets found: {len(doublets)}")

# ── Test Theorem 6.1 on each doublet ──
n_show = min(10, len(doublets))
print(f"\nTesting density mirror theorem on first {n_show} doublets:")
print("=" * 90)
print(f"{'#':>3}  {'E':>10}  {'dE_pair':>10}  "
      f"{'Mx:|<o|My|e>|':>13}  {'Mx:dens_err':>11}  "
      f"{'My:|<o|Mx|e>|':>13}  {'My:dens_err':>11}")
print("-" * 100)

# ── Collect results for summary figure ──
all_overlap = []
all_density_err = []
all_overlap_y = []
all_density_err_y = []
first_phi_even = None
first_phi_odd = None

for idx_d, (i1, i2) in enumerate(doublets[:n_show]):
    E1 = eigenvalues[i1]

    # Extract eigenstate as spinor (2, n, n)
    phi1 = eigenvectors[:, i1].reshape(2, n, n)

    if i2 is not None:
        E2 = eigenvalues[i2]
        dE = abs(E2 - E1)
    else:
        dE = np.nan

    # Apply Mx to get Kramers partner projection
    Mx_phi1 = apply_Mx(phi1)

    # Project onto Mx-even and Mx-odd
    phi_even = phi1 + Mx_phi1
    phi_odd  = phi1 - Mx_phi1

    norm_even = np.sqrt(np.sum(density(phi_even)) * dx_val**2)
    norm_odd  = np.sqrt(np.sum(density(phi_odd))  * dx_val**2)

    if norm_even < 1e-10 or norm_odd < 1e-10:
        print(f"{idx_d:3d}  {E1:10.4f}  {dE:10.2e}  {'(degenerate projection failed)':>40}")
        # Try using the second eigenstate to form a better basis
        if i2 is not None:
            phi2 = eigenvectors[:, i2].reshape(2, n, n)
            # Use phi2 as the odd partner
            Mx_phi2 = apply_Mx(phi2)
            phi_even2 = phi2 + Mx_phi2
            phi_odd2  = phi2 - Mx_phi2
            ne2 = np.sqrt(np.sum(density(phi_even2)) * dx_val**2)
            no2 = np.sqrt(np.sum(density(phi_odd2))  * dx_val**2)
            # Mix: use whichever has larger even/odd norms
            if ne2 > 1e-10 and no2 > 1e-10:
                phi_even = phi_even2
                phi_odd  = phi_odd2
                norm_even = ne2
                norm_odd  = no2
            else:
                # Both states are Mx eigenstates already: one even, one odd
                # phi1 is even (phi_odd ~ 0), phi2 is odd (phi_even2 ~ 0) or vice versa
                if norm_even > norm_odd:
                    phi_even = phi1
                    phi_odd  = phi2
                    norm_even = np.sqrt(np.sum(density(phi_even)) * dx_val**2)
                    norm_odd  = np.sqrt(np.sum(density(phi_odd))  * dx_val**2)
                else:
                    phi_even = phi2
                    phi_odd  = phi1
                    norm_even = np.sqrt(np.sum(density(phi_even)) * dx_val**2)
                    norm_odd  = np.sqrt(np.sum(density(phi_odd))  * dx_val**2)

    phi_even = phi_even / norm_even
    phi_odd  = phi_odd  / norm_odd

    # Test 1: |<φ_odd | My φ_even>|
    My_even = apply_My(phi_even)
    ovlp = overlap(phi_odd, My_even, dx_val)
    abs_ovlp = abs(ovlp)

    # Test 2: density mirror: ρ_even(x, -y) vs ρ_odd(x, y)
    rho_even = density(phi_even)
    rho_odd  = density(phi_odd)
    # Reflect ρ_even in y: row i → (n-i) % n
    rho_even_yflip = _fft_flip(rho_even, axis=0)
    density_err = np.sum(np.abs(rho_even_yflip - rho_odd)) / np.sum(rho_odd)

    # ── My projections (analogous test) ──
    My_phi1 = apply_My(phi1)
    phi_yeven = phi1 + My_phi1
    phi_yodd  = phi1 - My_phi1
    norm_ye = np.sqrt(np.sum(density(phi_yeven)) * dx_val**2)
    norm_yo = np.sqrt(np.sum(density(phi_yodd))  * dx_val**2)

    if norm_ye < 1e-10 or norm_yo < 1e-10:
        if i2 is not None:
            phi2 = eigenvectors[:, i2].reshape(2, n, n)
            My_phi2 = apply_My(phi2)
            phi_yeven2 = phi2 + My_phi2
            phi_yodd2  = phi2 - My_phi2
            nye2 = np.sqrt(np.sum(density(phi_yeven2)) * dx_val**2)
            nyo2 = np.sqrt(np.sum(density(phi_yodd2))  * dx_val**2)
            if nye2 > 1e-10 and nyo2 > 1e-10:
                phi_yeven, phi_yodd = phi_yeven2, phi_yodd2
                norm_ye, norm_yo = nye2, nyo2
            else:
                if norm_ye > norm_yo:
                    phi_yeven, phi_yodd = phi1, phi2
                else:
                    phi_yeven, phi_yodd = phi2, phi1
                norm_ye = np.sqrt(np.sum(density(phi_yeven)) * dx_val**2)
                norm_yo = np.sqrt(np.sum(density(phi_yodd))  * dx_val**2)

    phi_yeven = phi_yeven / norm_ye
    phi_yodd  = phi_yodd  / norm_yo

    # My test 1: |<φ_yodd | Mx φ_yeven>| (should be 1.0)
    Mx_yeven = apply_Mx(phi_yeven)
    ovlp_y = overlap(phi_yodd, Mx_yeven, dx_val)
    abs_ovlp_y = abs(ovlp_y)

    # My test 2: ρ_yeven(-x,y) vs ρ_yodd(x,y)
    rho_yeven = density(phi_yeven)
    rho_yodd  = density(phi_yodd)
    rho_yeven_xflip = _fft_flip(rho_yeven, axis=1)
    density_err_y = np.sum(np.abs(rho_yeven_xflip - rho_yodd)) / np.sum(rho_yodd)

    # Store results for figure
    all_overlap.append(abs_ovlp)
    all_density_err.append(density_err)
    all_overlap_y.append(abs_ovlp_y)
    all_density_err_y.append(density_err_y)
    if idx_d == 0:
        first_phi_even = phi_even.copy()
        first_phi_odd = phi_odd.copy()

    print(f"{idx_d:3d}  {E1:10.4f}  {dE:10.2e}  "
          f"{abs_ovlp:10.6f}  {density_err:10.2e}  "
          f"{abs_ovlp_y:10.6f}  {density_err_y:10.2e}")

print("=" * 100)
print("\nExpected: all overlaps ≈ 1.0, all density_err ≈ 0.0")

# ── Summary figure ──
import os
os.makedirs("figures", exist_ok=True)

extent = [-L/2, L/2, -L/2, L/2]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=200)

# Row 1: first Kramers doublet densities
rho_e = density(first_phi_even)
rho_o = density(first_phi_odd)
rho_e_yflip = _fft_flip(rho_e, axis=0)
diff = np.abs(rho_e_yflip - rho_o)

im0 = axes[0, 0].imshow(rho_e, origin="lower", extent=extent, cmap="inferno", aspect="equal")
axes[0, 0].set_title(r"$\rho_{\mathrm{even}}$ ($M_x = +1$)")
fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

im1 = axes[0, 1].imshow(rho_o, origin="lower", extent=extent, cmap="inferno", aspect="equal")
axes[0, 1].set_title(r"$\rho_{\mathrm{odd}}$ ($M_x = -1$)")
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

vmax_diff = diff.max() if diff.max() > 0 else 1e-10
im2 = axes[0, 2].imshow(diff, origin="lower", extent=extent, cmap="RdBu_r", aspect="equal",
                         vmin=0, vmax=vmax_diff)
axes[0, 2].set_title(r"$| \rho_{\mathrm{even}}(x,-y) - \rho_{\mathrm{odd}}(x,y) |$")
fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

# Row 2: bar charts (use first 2 panels, hide 3rd)
n_bars = len(all_overlap)
x_bars = np.arange(n_bars)

axes[1, 0].bar(x_bars, all_overlap, color="steelblue", edgecolor="k", linewidth=0.5)
axes[1, 0].axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="ideal = 1.0")
axes[1, 0].set_xlabel("Doublet index")
axes[1, 0].set_ylabel(r"$|\langle\phi_{\mathrm{odd}}|M_y|\phi_{\mathrm{even}}\rangle|$")
axes[1, 0].set_title(r"$|\langle\phi_{\mathrm{odd}}|M_y\phi_{\mathrm{even}}\rangle|$")
axes[1, 0].set_xticks(x_bars)
axes[1, 0].legend(fontsize=9)
axes[1, 0].set_ylim(0, 1.15)

axes[1, 1].bar(x_bars, all_density_err, color="coral", edgecolor="k", linewidth=0.5)
axes[1, 1].set_yscale("log")
axes[1, 1].set_xlabel("Doublet index")
axes[1, 1].set_ylabel("Density mirror error")
axes[1, 1].set_title("Density mirror error (log scale)")
axes[1, 1].set_xticks(x_bars)

axes[1, 2].axis("off")

fig.suptitle("Theorem 6.1 verification on exact eigenstates (n=32, harmonic ellipse)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/fig_exact_diag_theorem.png")
print(f"\nFigure saved to figures/fig_exact_diag_theorem.png")
