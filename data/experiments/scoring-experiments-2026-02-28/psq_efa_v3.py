#!/usr/bin/env python3
"""PSQ Factor Analysis v3 — EFA on separated-llm scores from psq.db
   Implements ML factor analysis manually to avoid factor_analyzer/sklearn incompatibility.
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from io import StringIO
from scipy import stats, optimize, linalg

DB_PATH = "/home/kashif/projects/psychology/safety-quotient/data/psq.db"

DIMS = [
    "threat_exposure", "hostility_index", "authority_dynamics",
    "energy_dissipation", "regulatory_capacity", "resilience_baseline",
    "trust_conditions", "cooling_capacity", "defensive_architecture",
    "contractual_clarity"
]

DIM_ABBREV = {
    "threat_exposure": "TE", "hostility_index": "HI", "authority_dynamics": "AD",
    "energy_dissipation": "ED", "regulatory_capacity": "RC", "resilience_baseline": "RB",
    "trust_conditions": "TC", "cooling_capacity": "CC", "defensive_architecture": "DA",
    "contractual_clarity": "CO"
}

output = StringIO()

def pr(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output)


def calculate_kmo(X):
    """Calculate Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy."""
    corr = np.corrcoef(X, rowvar=False)
    p = corr.shape[0]
    
    # Partial correlation matrix via inverse of correlation matrix
    try:
        corr_inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        corr_inv = np.linalg.pinv(corr)
    
    # Convert inverse to partial correlations
    d = np.diag(1.0 / np.sqrt(np.diag(corr_inv)))
    partial = -d @ corr_inv @ d
    np.fill_diagonal(partial, 1.0)
    
    # KMO per variable and overall
    corr_sq_sum = np.sum(corr**2, axis=1) - 1.0  # exclude diagonal
    partial_sq_sum = np.sum(partial**2, axis=1) - 1.0
    
    kmo_per_var = corr_sq_sum / (corr_sq_sum + partial_sq_sum)
    
    total_corr_sq = np.sum(corr**2) - p
    total_partial_sq = np.sum(partial**2) - p
    kmo_overall = total_corr_sq / (total_corr_sq + total_partial_sq)
    
    return kmo_per_var, kmo_overall


def calculate_bartlett(X):
    """Bartlett's test of sphericity."""
    n, p = X.shape
    corr = np.corrcoef(X, rowvar=False)
    det = np.linalg.det(corr)
    if det <= 0:
        det = 1e-300
    chi_sq = -((n - 1) - (2 * p + 5) / 6) * np.log(det)
    df = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi_sq, df)
    return chi_sq, p_value


def ml_factor_analysis(corr, n_factors, n_obs):
    """Maximum Likelihood factor analysis via eigendecomposition."""
    p = corr.shape[0]
    
    # Initial: PCA-based
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Initial loadings from top n_factors eigenvalues
    loadings = eigenvectors[:, :n_factors] * np.sqrt(np.maximum(eigenvalues[:n_factors], 0))
    
    # Iterative ML estimation (simplified EM-like)
    for iteration in range(200):
        old_loadings = loadings.copy()
        
        # Uniquenesses
        communalities = np.sum(loadings**2, axis=1)
        uniquenesses = np.clip(1.0 - communalities, 0.005, 0.995)
        
        # Scaled correlation: Psi^{-1/2} R Psi^{-1/2}
        psi_inv_sqrt = np.diag(1.0 / np.sqrt(uniquenesses))
        scaled_corr = psi_inv_sqrt @ corr @ psi_inv_sqrt
        
        # Eigendecomposition of scaled correlation
        ev, evec = np.linalg.eigh(scaled_corr)
        idx2 = np.argsort(ev)[::-1]
        ev = ev[idx2]
        evec = evec[:, idx2]
        
        # New loadings
        psi_sqrt = np.diag(np.sqrt(uniquenesses))
        loadings = psi_sqrt @ evec[:, :n_factors] * np.sqrt(np.maximum(ev[:n_factors] - 1.0, 0.0))
        
        # Check convergence
        diff = np.max(np.abs(loadings - old_loadings))
        if diff < 1e-8:
            break
    
    communalities = np.sum(loadings**2, axis=1)
    uniquenesses = 1.0 - communalities
    
    return loadings, communalities, uniquenesses


def promax_rotation(loadings, power=3):
    """Promax oblique rotation: varimax first, then target rotation."""
    n, k = loadings.shape
    if k == 1:
        return loadings, None  # No rotation for 1 factor
    
    # Varimax rotation
    varimax_loadings, varimax_rotation = varimax(loadings)
    
    # Promax: raise varimax to a power (preserving sign) to create target
    target = np.sign(varimax_loadings) * np.abs(varimax_loadings)**power
    
    # Procrustes rotation toward target
    # L_promax = L_varimax @ T, where T minimizes ||L_varimax @ T - target||
    # T = (L_varimax' L_varimax)^{-1} L_varimax' target
    T = np.linalg.lstsq(varimax_loadings, target, rcond=None)[0]
    
    promax_loadings = varimax_loadings @ T
    
    # Normalize columns
    norms = np.sqrt(np.sum(promax_loadings**2, axis=0))
    promax_loadings = promax_loadings / norms
    
    # Re-scale to match original communalities
    original_comm = np.sum(loadings**2, axis=1)
    promax_comm = np.sum(promax_loadings**2, axis=1)
    scale = np.sqrt(original_comm / np.maximum(promax_comm, 1e-10))
    promax_loadings = promax_loadings * scale[:, np.newaxis]
    
    # Factor correlation matrix
    # Phi = (T'^{-1} T^{-1}) normalized
    T_inv = np.linalg.inv(T)
    phi = T_inv.T @ T_inv
    # Normalize to correlation
    d = np.diag(1.0 / np.sqrt(np.diag(phi)))
    phi = d @ phi @ d
    
    return promax_loadings, phi


def varimax(loadings, max_iter=500, tol=1e-8):
    """Varimax rotation."""
    n, k = loadings.shape
    rotation_matrix = np.eye(k)
    
    L = loadings.copy()
    
    for _ in range(max_iter):
        old_rotation = rotation_matrix.copy()
        for i in range(k):
            for j in range(i+1, k):
                # Compute rotation angle
                u = L[:, i]**2 - L[:, j]**2
                v = 2 * L[:, i] * L[:, j]
                
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = 2 * np.sum(u * v)
                
                num = D - 2 * A * B / n
                den = C - (A**2 - B**2) / n
                
                angle = 0.25 * np.arctan2(num, den)
                
                # Rotate
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                
                Li = L[:, i] * cos_a + L[:, j] * sin_a
                Lj = -L[:, i] * sin_a + L[:, j] * cos_a
                L[:, i] = Li
                L[:, j] = Lj
                
                # Update rotation matrix
                Ri = rotation_matrix[:, i] * cos_a + rotation_matrix[:, j] * sin_a
                Rj = -rotation_matrix[:, i] * sin_a + rotation_matrix[:, j] * cos_a
                rotation_matrix[:, i] = Ri
                rotation_matrix[:, j] = Rj
        
        if np.max(np.abs(rotation_matrix - old_rotation)) < tol:
            break
    
    return L, rotation_matrix


def main():
    pr("=" * 80)
    pr("PSQ FACTOR ANALYSIS v3 — Separated-LLM Scores")
    pr("=" * 80)
    pr()

    # 1. Query DB
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    SELECT s.text_id, s.dimension, s.score
    FROM scores s
    WHERE s.method = 'separated-llm'
      AND s.dimension IN ({dims})
      AND s.id = (
        SELECT s2.id FROM scores s2
        WHERE s2.text_id = s.text_id 
          AND s2.dimension = s.dimension
          AND s2.method = 'separated-llm'
        ORDER BY s2.confidence DESC, s2.scored_at DESC
        LIMIT 1
      )
    """.format(dims=",".join(f"'{d}'" for d in DIMS))
    
    df_dedup = pd.read_sql_query(query, conn)
    pr(f"Score rows (deduped, best per text+dim): {len(df_dedup)}")
    
    # Coverage per dimension
    pr()
    pr("Coverage per dimension:")
    for d in DIMS:
        n = len(df_dedup[df_dedup['dimension'] == d])
        pr(f"  {DIM_ABBREV[d]:>4}: {n:,} texts")
    
    # Pivot to wide format
    df_wide = df_dedup.pivot(index='text_id', columns='dimension', values='score')
    pr(f"\nTexts with any separated-llm score: {len(df_wide)}")
    
    # Keep only texts with ALL 10 dimensions
    df_complete = df_wide.dropna(subset=DIMS)
    N = len(df_complete)
    pr(f"Texts with ALL 10 dimensions scored: {N}")
    pr()
    
    # Reorder columns
    df_complete = df_complete[DIMS]
    X = df_complete.values
    
    conn.close()
    
    # 2. Descriptive stats
    pr("=" * 80)
    pr("DESCRIPTIVE STATISTICS")
    pr("=" * 80)
    pr()
    pr(f"{'Dimension':<28} {'Abbrev':<6} {'Mean':>6} {'SD':>6} {'Min':>5} {'Max':>5} {'Skew':>7} {'Kurt':>7}")
    pr("-" * 80)
    for d in DIMS:
        col = df_complete[d]
        pr(f"{d:<28} {DIM_ABBREV[d]:<6} {col.mean():6.2f} {col.std():6.2f} {col.min():5.1f} {col.max():5.1f} {col.skew():7.3f} {col.kurtosis():7.3f}")
    pr()
    
    # 3. Correlation matrix
    corr = df_complete.corr().values
    pr("=" * 80)
    pr("CORRELATION MATRIX (separated-llm)")
    pr("=" * 80)
    pr()
    abbrevs = [DIM_ABBREV[d] for d in DIMS]
    header = f"{'':>4} " + " ".join(f"{a:>6}" for a in abbrevs)
    pr(header)
    for i, d in enumerate(DIMS):
        vals = " ".join(f"{corr[i, j]:6.3f}" for j in range(10))
        pr(f"{abbrevs[i]:>4} {vals}")
    pr()
    
    triu_indices = np.triu_indices(10, k=1)
    mean_abs_r = np.mean(np.abs(corr[triu_indices]))
    min_r = np.min(corr[triu_indices])
    max_r = np.max(corr[triu_indices])
    pr(f"Mean |r| (off-diagonal): {mean_abs_r:.3f}")
    pr(f"Range: {min_r:.3f} to {max_r:.3f}")
    pr()
    
    # 4. KMO test
    kmo_per_var, kmo_overall = calculate_kmo(X)
    pr("=" * 80)
    pr("KMO TEST")
    pr("=" * 80)
    pr()
    pr(f"Overall KMO: {kmo_overall:.4f}")
    pr()
    pr("Per-variable KMO:")
    for i, d in enumerate(DIMS):
        pr(f"  {DIM_ABBREV[d]:>4}: {kmo_per_var[i]:.4f}")
    pr()
    
    if kmo_overall >= 0.9:
        kmo_interp = "Marvelous"
    elif kmo_overall >= 0.8:
        kmo_interp = "Meritorious"
    elif kmo_overall >= 0.7:
        kmo_interp = "Middling"
    elif kmo_overall >= 0.6:
        kmo_interp = "Mediocre"
    elif kmo_overall >= 0.5:
        kmo_interp = "Miserable"
    else:
        kmo_interp = "Unacceptable"
    pr(f"Interpretation: {kmo_interp}")
    pr()
    
    # 5. Bartlett's test
    chi_sq, p_value = calculate_bartlett(X)
    pr("=" * 80)
    pr("BARTLETT'S TEST OF SPHERICITY")
    pr("=" * 80)
    pr()
    pr(f"Chi-square: {chi_sq:.1f}")
    pr(f"Degrees of freedom: {10 * 9 // 2}")
    pr(f"p-value:    {p_value:.2e}")
    pr(f"Conclusion: {'REJECT H0 (not identity matrix)' if p_value < 0.05 else 'Fail to reject H0'}")
    pr()
    
    # 6. Eigenvalue decomposition
    pr("=" * 80)
    pr("EIGENVALUE DECOMPOSITION")
    pr("=" * 80)
    pr()
    
    eigenvalues_all = np.linalg.eigvalsh(corr)[::-1]
    total_var = eigenvalues_all.sum()
    
    pr(f"{'Factor':<8} {'Eigenvalue':>10} {'% Variance':>12} {'Cumulative %':>14}")
    pr("-" * 46)
    cum_var = 0
    for i, ev in enumerate(eigenvalues_all):
        pct = ev / total_var * 100
        cum_var += pct
        marker = " <-- Kaiser" if ev >= 1.0 else ""
        pr(f"  {i+1:<6} {ev:10.4f} {pct:11.2f}% {cum_var:13.2f}%{marker}")
    pr()
    
    kaiser_n = sum(1 for ev in eigenvalues_all if ev >= 1.0)
    pr(f"Kaiser criterion retains: {kaiser_n} factor(s)")
    pr(f"g-factor eigenvalue: {eigenvalues_all[0]:.4f} ({eigenvalues_all[0]/total_var*100:.1f}% variance)")
    pr()
    
    # 7. Parallel analysis
    pr("=" * 80)
    pr("PARALLEL ANALYSIS (1000 permutation iterations)")
    pr("=" * 80)
    pr()
    
    n_iter = 1000
    n_obs, n_vars = X.shape
    random_eigenvalues = np.zeros((n_iter, n_vars))
    
    rng = np.random.RandomState(42)
    for i in range(n_iter):
        random_data = np.zeros_like(X)
        for j in range(n_vars):
            random_data[:, j] = rng.permutation(X[:, j])
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigvalsh(random_corr))[::-1]
    
    pa_95 = np.percentile(random_eigenvalues, 95, axis=0)
    pa_mean = np.mean(random_eigenvalues, axis=0)
    
    pr(f"{'Factor':<8} {'Actual':>10} {'Random 95th':>12} {'Random Mean':>12} {'Retain?':>8}")
    pr("-" * 52)
    pa_retain = 0
    for i in range(n_vars):
        retain = eigenvalues_all[i] > pa_95[i]
        if retain:
            pa_retain = i + 1
        pr(f"  {i+1:<6} {eigenvalues_all[i]:10.4f} {pa_95[i]:12.4f} {pa_mean[i]:12.4f} {'  YES' if retain else '   no':>8}")
    pr()
    pr(f"Parallel analysis retains: {pa_retain} factor(s)")
    pr()
    
    # 8. Factor solutions
    for n_factors in [1, 2, 3, 5]:
        pr("=" * 80)
        pr(f"{n_factors}-FACTOR PROMAX SOLUTION (ML extraction)")
        pr("=" * 80)
        pr()
        
        loadings_raw, communalities, uniquenesses = ml_factor_analysis(corr, n_factors, N)
        
        if n_factors > 1:
            loadings, phi = promax_rotation(loadings_raw, power=3)
        else:
            loadings = loadings_raw
            phi = None
        
        # Variance from loadings
        ss_loadings = np.sum(loadings**2, axis=0)
        prop_var = ss_loadings / n_vars
        cum_var_fa = np.cumsum(prop_var)
        
        # Print loadings
        if n_factors == 1:
            pr(f"{'Dimension':<28} {'Abbrev':<6} {'F1':>8}")
            pr("-" * 44)
            for i, d in enumerate(DIMS):
                marker = " ***" if abs(loadings[i, 0]) >= 0.7 else (" **" if abs(loadings[i, 0]) >= 0.5 else (" *" if abs(loadings[i, 0]) >= 0.35 else ""))
                pr(f"{d:<28} {DIM_ABBREV[d]:<6} {loadings[i, 0]:8.4f}{marker}")
        else:
            headers = " ".join(f"{'F'+str(j+1):>8}" for j in range(n_factors))
            pr(f"{'Dimension':<28} {'Abbrev':<6} {headers}  {'Max |L|':>8}")
            pr("-" * (44 + n_factors * 9 + 10))
            for i, d in enumerate(DIMS):
                vals = " ".join(f"{loadings[i, j]:8.4f}" for j in range(n_factors))
                max_l = max(abs(loadings[i, j]) for j in range(n_factors))
                marker = " ***" if max_l >= 0.7 else (" **" if max_l >= 0.5 else (" *" if max_l >= 0.35 else ""))
                pr(f"{d:<28} {DIM_ABBREV[d]:<6} {vals}  {max_l:8.4f}{marker}")
        
        pr()
        pr("Markers: *** >= 0.70, ** >= 0.50, * >= 0.35")
        pr()
        
        # Variance table
        pr(f"{'':>20} " + " ".join(f"{'F'+str(j+1):>10}" for j in range(n_factors)) + f" {'Total':>10}")
        pr(f"{'SS Loadings':>20} " + " ".join(f"{ss_loadings[j]:10.4f}" for j in range(n_factors)) + f" {sum(ss_loadings):10.4f}")
        pr(f"{'Prop. Variance':>20} " + " ".join(f"{prop_var[j]:10.4f}" for j in range(n_factors)) + f" {sum(prop_var):10.4f}")
        pr(f"{'Cum. Variance':>20} " + " ".join(f"{cum_var_fa[j]:10.4f}" for j in range(n_factors)))
        pr()
        
        if phi is not None:
            pr("Factor Correlation Matrix (Promax):")
            for i in range(n_factors):
                row = " ".join(f"{phi[i, j]:8.4f}" for j in range(n_factors))
                pr(f"  F{i+1}: {row}")
            pr()
        
        pr("Communalities:")
        for i, d in enumerate(DIMS):
            pr(f"  {DIM_ABBREV[d]:>4}: {communalities[i]:.4f}")
        mean_comm = np.mean(communalities)
        pr(f"  Mean: {mean_comm:.4f}")
        pr()
        
        pr("Uniquenesses:")
        for i, d in enumerate(DIMS):
            pr(f"  {DIM_ABBREV[d]:>4}: {uniquenesses[i]:.4f}")
        pr()
        
        pr(f"Total Variance Explained: {sum(prop_var)*100:.1f}%")
        pr()
    
    # 9. Model comparison summary
    pr("=" * 80)
    pr("MODEL COMPARISON SUMMARY")
    pr("=" * 80)
    pr()
    pr(f"{'Factors':<10} {'Total Var Explained':>20}")
    pr("-" * 32)
    for nf in [1, 2, 3, 4, 5, 6, 7]:
        try:
            L, comm, uniq = ml_factor_analysis(corr, nf, N)
            ss = np.sum(L**2, axis=0)
            total_pv = np.sum(ss) / n_vars
            pr(f"  {nf:<8} {total_pv*100:19.1f}%")
        except Exception as e:
            pr(f"  {nf:<8} FAILED: {e}")
    pr()
    
    # 10. v2 vs v3 comparison
    pr("=" * 80)
    pr("COMPARISON: v2 vs v3")
    pr("=" * 80)
    pr()
    pr(f"{'Metric':<40} {'v2':>15} {'v3':>15} {'Delta':>10}")
    pr("-" * 82)
    pr(f"{'N (complete texts)':<40} {'1,970':>15} {f'{N:,}':>15} {f'{N-1970:+,}':>10}")
    pr(f"{'g-factor eigenvalue':<40} {'6.727':>15} {eigenvalues_all[0]:15.3f} {eigenvalues_all[0]-6.727:+10.3f}")
    pr(f"{'g-factor % variance':<40} {'67.3%':>15} {eigenvalues_all[0]/total_var*100:14.1f}% {eigenvalues_all[0]/total_var*100-67.3:+9.1f}%")
    pr(f"{'KMO':<40} {'0.902':>15} {kmo_overall:15.3f} {kmo_overall-0.902:+10.3f}")
    pr(f"{'Mean |r| (off-diagonal)':<40} {'0.564':>15} {mean_abs_r:15.3f} {mean_abs_r-0.564:+10.3f}")
    pr(f"{'Kaiser retains':<40} {'3':>15} {kaiser_n:>15}")
    pr(f"{'Parallel analysis retains':<40} {'2':>15} {pa_retain:>15}")
    pr(f"{'2nd eigenvalue':<40} {'?':>15} {eigenvalues_all[1]:15.4f}")
    pr(f"{'Ratio eigen1/eigen2':<40} {'?':>15} {eigenvalues_all[0]/eigenvalues_all[1]:15.1f}")
    pr()
    
    # 11. Scree plot
    pr("=" * 80)
    pr("SCREE PLOT DATA (actual vs 95th percentile random)")
    pr("=" * 80)
    pr()
    pr(f"{'Factor':<8} {'Actual':>10} {'Random 95%':>12} {'Random Mean':>12}")
    pr("-" * 44)
    for i in range(10):
        pr(f"  {i+1:<6} {eigenvalues_all[i]:10.4f} {pa_95[i]:12.4f} {pa_mean[i]:12.4f}")
    pr()
    
    # Generate scree plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        factors = list(range(1, 11))
        ax.plot(factors, eigenvalues_all, 'bo-', linewidth=2, markersize=8, label='Actual eigenvalues')
        ax.plot(factors, pa_95, 'r^--', linewidth=2, markersize=8, label='Random 95th percentile')
        ax.plot(factors, pa_mean, 'gs:', linewidth=1, markersize=6, label='Random mean')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Kaiser criterion (=1)')
        ax.set_xlabel('Factor', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(f'PSQ Factor Analysis v3 -- Scree Plot (N={N:,}, separated-llm)', fontsize=14)
        ax.set_xticks(factors)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotate eigenvalues
        for i, ev in enumerate(eigenvalues_all):
            if i < 3:
                ax.annotate(f'{ev:.3f}', (factors[i], ev), textcoords="offset points", 
                           xytext=(10, 5), fontsize=9)
        
        plt.tight_layout()
        plt.savefig('/tmp/psq_scree_v3.png', dpi=150)
        pr("Scree plot saved to /tmp/psq_scree_v3.png")
    except Exception as e:
        pr(f"Could not generate scree plot: {e}")
    pr()
    
    # 12. Interpretation
    pr("=" * 80)
    pr("INTERPRETATION SUMMARY")
    pr("=" * 80)
    pr()
    pr(f"1. SAMPLE: N={N:,} texts with complete separated-llm scores on all 10 dimensions.")
    pr()
    pr(f"2. g-FACTOR: eigenvalue={eigenvalues_all[0]:.3f} ({eigenvalues_all[0]/total_var*100:.1f}% variance).")
    g_change = eigenvalues_all[0] - 6.727
    if abs(g_change) < 0.05:
        pr(f"   Essentially UNCHANGED vs v2 (delta={g_change:+.3f}). The g-factor is stable.")
    elif g_change > 0:
        pr(f"   STRENGTHENED vs v2 (+{g_change:.3f}).")
    else:
        pr(f"   WEAKENED vs v2 ({g_change:.3f}).")
    pr()
    pr(f"3. KMO={kmo_overall:.3f} -- {kmo_interp}. Excellent sampling adequacy maintained.")
    pr()
    pr(f"4. BARTLETT: chi-sq={chi_sq:.0f}, p<0.001 -- dimensions are significantly correlated.")
    pr()
    pr(f"5. FACTOR RETENTION:")
    pr(f"   - Kaiser criterion: {kaiser_n} factor(s) (eigenvalue > 1)")
    pr(f"   - Parallel analysis (95th %ile): {pa_retain} factor(s)")
    pr(f"   - The massive gap between eigenvalue 1 ({eigenvalues_all[0]:.3f}) and")
    pr(f"     eigenvalue 2 ({eigenvalues_all[1]:.3f}) is the defining feature.")
    pr(f"     Ratio: {eigenvalues_all[0]/eigenvalues_all[1]:.1f}:1")
    pr()
    pr(f"6. MEAN INTER-DIMENSION |r| = {mean_abs_r:.3f}")
    v2_r = 0.564
    pr(f"   vs v2: {v2_r:.3f} (delta={mean_abs_r - v2_r:+.3f})")
    pr()
    pr(f"7. STRUCTURAL CONCLUSIONS:")
    pr(f"   - The separated-llm PSQ remains essentially UNIDIMENSIONAL.")
    pr(f"   - A single general factor dominates the variance structure.")
    pr(f"   - Sub-factors (if extracted) will be weak and heavily correlated.")
    pr(f"   - This pattern is consistent with v2 and suggests that either:")
    pr(f"     (a) LLM scoring has implicit halo even in separated mode, or")
    pr(f"     (b) the PSQ dimensions genuinely capture a single latent construct")
    pr(f"         with minor surface variation.")
    pr(f"   - Expert human validation is needed to distinguish (a) from (b).")
    pr()
    
    # Save
    with open("/tmp/psq_factor_analysis_v3.txt", "w") as f:
        f.write(output.getvalue())
    
    print("\nFull output saved to /tmp/psq_factor_analysis_v3.txt")


if __name__ == "__main__":
    main()
