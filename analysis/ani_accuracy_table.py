"""ANI-1ccx accuracy table for referee comment R2.5b.

Trains the two models specified in ``table.md`` on the fixed 85:5:10 split of
the 50k ANI-1ccx subset and reports:

  * MAE and RMSE over all 5,000 test conformers [kcal/mol]
  * MAE per atom, computed per molecule and averaged over molecules
  * RMAE (per molecule, averaged over molecules; global value also printed)
  * within-molecule relative-energy RMSE over the conformers of each molecule
  * test sample counts (conformers and molecules, mean conformers/molecule)

Models:
  * single-fidelity KRR on 2^6 = 64 CCSD(T)*/CBS labels
  * adaptive-MFML with terminal composition [768, 112, 28, 12]

Molecule identity is taken from the atomic-number sequence Z of each conformer,
because the prepared 50k npz does not retain the original molecule IDs.

Outputs are written to ``results/ani_accuracy_table/``.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from qml.kernels import matern_kernel
from qml.math import cho_solve
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from saturatedmfml.dataset_paths import ANI_DIR, RESULTS_ROOT, results_dir
from saturatedmfml.dataloaders import partition_ANI_data
from saturatedmfml.experiment_params import get_hyperparams
from saturatedmfml.MFML_Model import ModelMFML

ADAPTIVE_SIZES = np.asarray([768, 112, 28, 12])
KRR_NMAX = 6
SEED = 42
OUT = RESULTS_ROOT / 'ani_accuracy_table'


def test_ids_and_atom_counts(seed=SEED):
    """Reproduce the 90/10 test split and identify molecules by connectivity.

    The prepared 50k npz does not store molecule IDs.  Molecules are therefore
    identified by the canonical SMILES obtained from RDKit bond perception on
    each test geometry; structures for which perception fails are treated as
    separate molecules.
    """
    npz = np.load(f'{ANI_DIR}/ANI1x_multifidelity_50k_allfids.npz', allow_pickle=True)
    Z = npz['Z']
    R = npz['X']
    idx = np.arange(Z.shape[0])
    _, test_idx = train_test_split(idx, train_size=0.9, random_state=seed)
    Z_test, R_test = Z[test_idx], R[test_idx]

    mol_ids, n_atoms, n_fail = [], [], 0
    for k, (z, r) in enumerate(zip(Z_test, R_test)):
        n_atoms.append(len(z))
        try:
            mol = Chem.RWMol()
            for zz in z:
                mol.AddAtom(Chem.Atom(int(zz)))
            conf = Chem.Conformer(len(z))
            for i, xyz in enumerate(r):
                conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]),
                                                float(xyz[2])))
            mol.AddConformer(conf)
            rdDetermineBonds.DetermineBonds(mol, charge=0)
            mol_ids.append(Chem.MolToSmiles(mol))
        except Exception:
            mol_ids.append(f'FAIL_{k}')
            n_fail += 1
    return mol_ids, n_atoms, n_fail


def krr_sf_predictions():
    X_tr, _, X_te, y_tr, _, y_te = partition_ANI_data(seed=SEED, center=True)
    X_tr, y_tr = shuffle(X_tr, y_tr, random_state=SEED)
    n = 2 ** KRR_NMAX
    reg, sigma = get_hyperparams('ANI', 'ANI')
    K = matern_kernel(X_tr[:n], X_tr[:n], sigma=sigma, order=1, metric='l2')
    K[np.diag_indices_from(K)] += reg
    alpha = cho_solve(K, y_tr[:n, -1])
    K_test = matern_kernel(X_tr[:n], X_te, sigma=sigma, order=1, metric='l2')
    return np.dot(alpha, K_test), y_te[:, -1]


def adaptive_predictions():
    X_tr, X_val, X_te, y_tr, y_val, y_te = partition_ANI_data(seed=SEED, center=True)
    reg, sigma = get_hyperparams('ANI', 'ANI')
    index_file = Path(__file__).resolve().parent.parent / 'ANI_indexes.npy'
    if index_file.exists():
        index = np.copy(np.load(index_file, allow_pickle=True)[0][:, 0])
    else:
        index = np.arange(X_tr.shape[0])

    np.random.seed(SEED)
    s = ADAPTIVE_SIZES[0]
    select = np.random.choice(index, size=s, replace=False)
    X_sub = X_tr[select]
    y_trains = y_tr.T
    energies = np.zeros((y_trains.shape[0]), dtype=object)
    for i in range(y_trains.shape[0]):
        energies[i] = np.copy(y_trains[i, select])

    indexes = np.zeros((len(ADAPTIVE_SIZES)), dtype=object)
    patched = np.vstack([np.arange(s), np.arange(s)]).T
    for i in range(len(ADAPTIVE_SIZES)):
        indexes[i] = np.copy(patched)

    model = ModelMFML(reg=reg, kernel='matern', sigma=sigma, order=1,
                      metric='l2', gammas=None, p_bar=False)
    model.train(X_train_parent=X_sub, fidelities=None, y_trains=energies,
                indexes=indexes, shuffle=True, n_trains=ADAPTIVE_SIZES, seed=0)
    preds = model.predict(X_test=X_te, y_test=y_te[:, -1], optimiser='default')
    return preds, y_te[:, -1]


def metrics(pred, y, mol_ids, n_atoms):
    res = pred - y
    groups = defaultdict(list)
    for i, mid in enumerate(mol_ids):
        groups[mid].append(i)

    mae = float(np.mean(np.abs(res)))
    rmse = float(np.sqrt(np.mean(res ** 2)))
    rmae_global = float(np.mean(np.abs(res / y)))

    per_atom, rmae_mol, within = [], [], []
    for mid, ind in groups.items():
        n_at = n_atoms[ind[0]]
        r = res[ind]
        per_atom.append(np.mean(np.abs(r)) / n_at)
        rmae_mol.append(np.mean(np.abs(r / y[ind])))
        if len(ind) >= 2:
            yy = y[ind]
            pp = pred[ind]
            d = (yy - yy.min()) - (pp - pp.min())
            within.append(np.sqrt(np.mean(d ** 2)))

    n_conf = np.asarray([len(v) for v in groups.values()])
    return dict(
        n_test=len(y), n_mol=len(groups), mean_conf=float(n_conf.mean()),
        mae=mae, rmse=rmse, mae_atom=float(np.mean(per_atom)),
        rmae_mol=float(np.mean(rmae_mol)), rmae_global=rmae_global,
        within_rmse=float(np.mean(within)), n_mol_multi=len(within),
        n_conf_multi=int(sum(len(groups[k]) for k in groups
                             if len(groups[k]) >= 2)),
    )


def write_tables(rows, n_fail):
    results_dir(OUT)
    labels = [('Single-fidelity KRR ($2^6$)', rows[0]),
              ('Adaptive-MFML [768,112,28,12]', rows[1])]
    md = ['| Model | N_test | N_mol | MAE [kcal/mol] | RMSE [kcal/mol] | MAE per atom [kcal/mol] | RMAE | Within-molecule RMSE [kcal/mol] | mean n_conf |',
          '|---|---|---|---|---|---|---|---|---|']
    for name, m in labels:
        md.append(f'| {name} | {m["n_test"]} | {m["n_mol"]} | {m["mae"]:.4f} | '
                  f'{m["rmse"]:.4f} | {m["mae_atom"]:.4f} | {m["rmae_mol"]:.4f} | '
                  f'{m["within_rmse"]:.4f} | {m["mean_conf"]:.2f} |')
    md += ['', 'Molecule identity: canonical SMILES from RDKit connectivity; '
               f'{n_fail} structures with failed bond perception are '
               'treated as separate molecules. '
               f'Within-molecule RMSE uses molecules with $\\ge2$ test conformers '
               f'(N={rows[0]["n_mol_multi"]}, {rows[0]["n_conf_multi"]} conformers); '
               'single-conformer molecules are excluded. Global RMAE (Eq. 6): '
               f'KRR {rows[0]["rmae_global"]:.4f}, adaptive {rows[1]["rmae_global"]:.4f}.']
    (OUT / 'ani_accuracy_table.md').write_text('\n'.join(md) + '\n')

    tex = r'''\begin{table}[htb!]
  \centering
  \caption{Accuracy of the adaptive-MFML model and of the single-fidelity KRR
  model on the holdout test set of ANI-1ccx. Both models are evaluated at
  comparable total acquisition costs ($\sim$3.4~h and $\sim$2.8~h).
  Energies are mean-centred. MAE/RMSE are over all test conformers; the
  per-atom error and RMAE are computed per molecule and averaged over
  molecules. Molecules are identified by canonical SMILES from RDKit
  connectivity: $N_{\rm mol}=%d$, mean $n_{\rm conf}=%.2f$; %d structures for
  which bond perception failed are treated as separate molecules. The
  within-molecule RMSE is the error in relative energies
  $E_i-\min_j E_j$ over the conformers of each molecule, averaged over
  molecules with at least two conformers ($N=%d$, %d conformers).}
  \label{tab:ani_accuracy}
  \begin{tabular}{|l|c|c|c|c|c|c|}
    \hline
    \textbf{Model} & \textbf{$N_{\rm test}$} & \textbf{$N_{\rm mol}$}
    & \textbf{MAE} & \textbf{RMSE} & \textbf{MAE/atom} & \textbf{RMAE} \\
    & & & \textbf{[kcal/mol]} & \textbf{[kcal/mol]}
    & \textbf{[kcal/mol]} & \\
    \hline
    \hline
''' % (rows[0]['n_mol'], rows[0]['mean_conf'], n_fail, rows[0]['n_mol_multi'], rows[0]['n_conf_multi'])
    for name, m in labels:
        tex += (f'    {name.replace("$2^6$", "$2^6$")} & {m["n_test"]:,} & {m["n_mol"]} '
                f'& {m["mae"]:.3f} & {m["rmse"]:.3f} & {m["mae_atom"]:.3f} '
                f'& {m["rmae_mol"]:.4f} \\\\\n')
    tex += (r'''    \hline
    \multicolumn{7}{|l|}{\textbf{Within-molecule relative energies}
      \quad RMSE [kcal/mol]: KRR: %.3f \quad adaptive: %.3f
      \quad ($\bar n_{\rm conf}$: %.2f)} \\
    \hline
  \end{tabular}
\end{table}
''' % (rows[0]['within_rmse'], rows[1]['within_rmse'], rows[0]['mean_conf']))
    (OUT / 'ani_accuracy_table.tex').write_text(tex)
    print('\n'.join(md))


def main():
    mol_ids, n_atoms, n_fail = test_ids_and_atom_counts()
    print('training single-fidelity KRR (2^6)...', flush=True)
    pred_krr, y_krr = krr_sf_predictions()
    m_krr = metrics(pred_krr, y_krr, mol_ids, n_atoms)
    print('training adaptive-MFML [768,112,28,12]...', flush=True)
    pred_ad, y_ad = adaptive_predictions()
    m_ad = metrics(pred_ad, y_ad, mol_ids, n_atoms)
    write_tables([m_krr, m_ad], n_fail)
    print('wrote', OUT)
    for name, m in [('KRR', m_krr), ('adaptive', m_ad)]:
        print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()})


if __name__ == '__main__':
    main()
