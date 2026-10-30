"""ANI-1ccx accuracy table for referee comment R2.5b.

Trains the two models specified in ``table.md`` on the ANI-1ccx 50k subset
using an 85:5:10 train:validation:test split (seed 0; the validation split is
not used) and reports MAE, RMSE, per-atom MAE and RMAE on the 10 % test split.

Models:
  * single-fidelity KRR on 2^6 = 64 CCSD(T)*/CBS labels
  * adaptive-MFML with terminal composition [768, 112, 28, 12]

Molecule identity for the per-molecule averaging of per-atom error and RMAE is
taken from canonical SMILES obtained by RDKit bond perception, because the
prepared 50k npz does not retain the original molecule IDs.

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
SEED = 0
OUT = RESULTS_ROOT / 'ani_accuracy_table'


def test_ids_and_atom_counts(seed=SEED):
    """Reproduce the 85:5:10 split and identify test molecules by connectivity."""
    npz = np.load(f'{ANI_DIR}/ANI1x_multifidelity_50k_allfids.npz', allow_pickle=True)
    Z, R = npz['Z'], npz['X']
    idx = np.arange(Z.shape[0])
    # partition_ANI_data uses 90/10 first, then 85/90 of the train part;
    # the held-out test set is therefore the 10 % part (train 85 %, val 5 %).
    _, test_idx = train_test_split(idx, train_size=0.9, random_state=seed)
    Z_test, R_test = Z[test_idx], R[test_idx]

    mol_ids, n_atoms = [], []
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
    return mol_ids, n_atoms


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
    per_atom = []
    for _, ind in groups.items():
        r = res[ind]
        per_atom.append(np.mean(np.abs(r)) / n_atoms[ind[0]])
    rmae = float(np.mean(np.abs(res / y)))   # Eq. (6), over all test conformers
    return dict(mae=mae, rmse=rmse, mae_atom=float(np.mean(per_atom)), rmae=rmae)


def write_tables(rows):
    results_dir(OUT)
    labels = [('Single-fidelity KRR ($2^6$)', rows[0]),
              ('Adaptive-MFML [768,112,28,12]', rows[1])]
    md = ['| Model | MAE [kcal/mol] | RMSE [kcal/mol] | MAE per atom [kcal/mol] | RMAE |',
          '|---|---|---|---|---|']
    for name, m in labels:
        md.append(f'| {name} | {m["mae"]:.4f} | {m["rmse"]:.4f} | '
                  f'{m["mae_atom"]:.4f} | {m["rmae"]:.4f} |')
    (OUT / 'ani_accuracy_table.md').write_text('\n'.join(md) + '\n')

    caption = (
        r'\caption{Accuracy of the single-fidelity KRR and adaptive-MFML models '
        r'on the holdout test set of ANI-1ccx (seed 0, 85:5:10 split; the '
        r'validation split is not used here). Both models are evaluated at '
        r'comparable total acquisition costs ($\sim$2.8~h and $\sim$3.5~h, including two validation labels). '
        r'Energies are mean-centred; MAE, RMSE and RMAE are over all test '
        r'conformers (RMAE as in Eq. (6)); the per-atom error is computed per '
        r'molecule and averaged over molecules.}'
    )
    tex = [
        r'\begin{table}[htb!]',
        r'  \centering',
        '  ' + caption,
        r'  \label{tab:ani_accuracy}',
        r'  \begin{tabular}{|l|c|c|c|c|}',
        r'    \hline',
        r'    \textbf{Model} & \textbf{MAE [kcal/mol]} & \textbf{RMSE [kcal/mol]}'
        r' & \textbf{MAE/atom [kcal/mol]} & \textbf{RMAE} \\',
        r'    \hline',
        r'    \hline',
    ]
    for name, m in labels:
        tex.append(f'    {name} & {m["mae"]:.3f} & {m["rmse"]:.3f} & '
                   f'{m["mae_atom"]:.3f} & {m["rmae"]:.4f} \\\\')
    tex += [r'    \hline', r'  \end{tabular}', r'\end{table}', '']
    (OUT / 'ani_accuracy_table.tex').write_text('\n'.join(tex))
    print('\n'.join(md))


def main():
    mol_ids, n_atoms = test_ids_and_atom_counts()
    print('training single-fidelity KRR (2^6)...', flush=True)
    pred_krr, y_krr = krr_sf_predictions()
    m_krr = metrics(pred_krr, y_krr, mol_ids, n_atoms)
    print('training adaptive-MFML [768,112,28,12]...', flush=True)
    pred_ad, y_ad = adaptive_predictions()
    m_ad = metrics(pred_ad, y_ad, mol_ids, n_atoms)
    write_tables([m_krr, m_ad])
    print('wrote', OUT)
    for name, m in [('KRR', m_krr), ('adaptive', m_ad)]:
        print(name, {k: round(v, 5) for k, v in m.items()})


if __name__ == '__main__':
    main()
