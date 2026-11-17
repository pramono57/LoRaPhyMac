# APC closed-loop verification

This contracted one-run experiment supplies the evidence for the Paper 1
closed-loop APC behavior-verification section. It is exactly one frozen point
of the APC terrain study:

- 100 nodes and 2 relays;
- 3600-second horizon;
- terrain irregularity 90 m;
- seed 42;
- adaptive node and relay PTx;
- relay forwarding channel [1].

The workflow validates both the configuration and completed evidence. A PASS
requires physical APC accounting to balance, packet and collision conservation
to hold, and both node and relay PTx updates plus margin histories to be
observed.

Run from the repository root:

    python -m loraphymac.experiments.apc_closed_loop_verification.run

Outputs are written below:

    artifacts/experiments/apc_closed_loop_verification/<run_id>/

The verification_report.json file is the machine-readable PASS marker. Figures
and maintained tables reuse the shared single-run diagnostic builders, so this
workflow does not duplicate the simulator or plotting implementation.

========= Manual ID =============
Fungsinya untuk menghasilkan evidence perilaku closed-loop APC yang dibutuhkan Paper 1.
Ini merupakan satu titik eksperimen yang dibekukan, yaitu:
- 100 node dan 2 relay;
- waktu simulasi 3600 detik;
- Δh = 90 m;
- seed 42;
- node adaptive PTx;
- relay adaptive PTx;
- channel forwarding relay [1].
Berbeda dengan single_run_exploration, folder ini memiliki kontrak. Konfigurasi tidak boleh diubah sembarangan jika hasilnya akan disebut sebagai APC closed-loop verification Paper 1.
Workflow memeriksa antara lain:
- update PTx node benar-benar terjadi;
- update PTx relay benar-benar terjadi;
- margin history tersedia;
- perhitungan event APC konsisten;
- konservasi packet dan collision;
- tabel dan gambar yang diwajibkan tersedia.
Jika semua pemeriksaan lolos, dihasilkan:
verification_report.json
Output:
artifacts/experiments/apc_closed_loop_verification/<run_id>/
