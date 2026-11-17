# Single-run exploration

This workflow is the flexible research sandbox used before an experiment is
frozen. It uses the same simulator engine and central validation as every other
run, but it does not apply a publication contract.

Edit a copy of configs/default_v1.json or override the dictionary in the thin
notebook. The maintained starting point uses 100 nodes, 3600 seconds,
adaptive/adaptive PTx, and relay forwarding channels [1, 2, 3, 4].

Run from the repository root:

    python -m loraphymac.experiments.single_run_exploration.run

Outputs are written to:

    artifacts/experiments/single_run_exploration/<run_id>/

These outputs are diagnostic evidence, not publication evidence. Once a
scenario becomes part of a paper, copy its scientific choices into a named,
versioned experiment contract rather than treating an exploratory run as
frozen after the fact.


=== Manual ID ========
Fungsinya sebagai sandbox penelitian untuk:
- mencoba perubahan parameter;
- memeriksa routing;
- melihat event dan tabel debug;
- memeriksa distribusi PTx;
- menguji penggunaan beberapa channel relay;
- melakukan eksplorasi awal sebelum skenario dijadikan eksperimen formal.

Konfigurasi awal saat ini:
- 100 node;
- 3600 detik;
- Δh = 90 m;
- seed 42;
- node dan relay adaptive PTx;
- channel TX relay [1, 2, 3, 4].

Eksperimen ini tidak memiliki kontrak publikasi. Artinya parameternya boleh diubah sesuai kebutuhan eksplorasi. Hasilnya tidak otomatis dianggap sebagai evidence Paper 1.
Output:
artifacts/experiments/single_run_exploration/<run_id>/