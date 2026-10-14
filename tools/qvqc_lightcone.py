"""Gates in the readout qubit's backward light cone for each Pareto-front circuit.

    cd sota/QuantumVQC && ../../.venv/bin/python ../../tools/qvqc_lightcone.py

Gates outside the light cone cannot change the prediction but still count in obj2.
Writes reports/qvqc_rag_run/lightcone.json.
"""
import importlib, sys, csv, json
sys.path.insert(0, '.')
def lightcone(qc, readout):
    live = set(readout); keep = 0
    for inst in reversed(qc.data):
        if inst.operation.name in ("barrier", "measure"): continue
        qs = {qc.find_bit(q).index for q in inst.qubits}
        if qs & live:
            keep += 1; live |= qs
    return keep, sorted(live)
front = [l.split('|')[1].strip() for l in open('../../reports/qvqc_rag_run/summary.md') if l.startswith('| xXx')]
out=[]
for g in ['__seed__'] + front:
    mod = importlib.import_module('seed_vqc' if g=='__seed__' else f'models.network_{g}')
    qc, xp, wp = mod.build_circuit()
    m = {} if g=='__seed__' else json.load(open(f'results/{g}_metrics.json'))
    k, live = lightcone(qc, mod.READOUT)
    n = sum(1 for i in qc.data if i.operation.name not in ('barrier','measure'))
    out.append((g, qc.num_qubits, n, k, live, m.get('val_accuracy'), m.get('test_accuracy')))
    print(g, qc.num_qubits, n, k, live, m.get('val_accuracy'), m.get('test_accuracy'))
json.dump(out, open('../../reports/qvqc_rag_run/lightcone.json','w'), indent=1)
