# APC Load Sweep

## Default: Standard V1.0

`python -m loraphymac.experiments.apc_load_sweep.run` defaults to
`configs/standard_v1.json` and the Standard `sweep_cohort_v1` lifecycle. The
production spec contains the paired AA/FF 120-run grid. Preview it without
reserving a cohort or running simulations:

```bash
PYTHONPATH=src python -m loraphymac.experiments.apc_load_sweep.run --dry-run
```

To execute the Standard production grid after the applicable release and
research gates are approved, select the Standard config explicitly:

```bash
PYTHONPATH=src python -m loraphymac.experiments.apc_load_sweep.run \
  src/loraphymac/experiments/apc_load_sweep/configs/standard_v1.json --fail-fast
```

This command runs 120 simulations; it is documented here and was not executed
for this compatibility change.

To select the two-run integration profile, pass `--profile pilot` and an
isolated output root outside repository `artifacts/`:

```bash
PYTHONPATH=src python -m loraphymac.experiments.apc_load_sweep.run \
  --profile pilot --output-root /private/tmp/loraphymac-load-pilot --dry-run
```

Pilot output is non-production integration evidence. Standard runs reserve a
new cohort and never resume or overwrite an existing cohort. Repeating a
Standard execution creates a new cohort. No Standard production 120-run sweep
was run during this release preparation.

## Legacy IJIES R1.7 full-factorial workflow

Passing one of the recognized legacy configs as the positional argument keeps
the historical `paper_artifacts_v1` runner and output layout. The production
launcher `tools/run_r17_production.sh` continues to use these positional
configs; its R1.7 parameters and output namespace are unchanged. Legacy runs
resume by default; pass `--no-resume` to disable resume. Dry-run previews the
grid without creating output directories.

The legacy design is FF and AA, four node counts × three reporting periods ×
five seeds, for 120 total runs split into two 60-run configurations. Its
scientific parameters and artifact namespace are separate from the Standard
cohort lifecycle.

### Legacy design and output

Grid: 50/100/150/200 node × interval 30/60/90 s × seed 30/36/42/48/56 × FF/AA.
Total 120 run, 60 pasangan FF–AA. Semua memakai Δh=90 m, durasi 3600 s,
SF7/BW125 kHz, payload 50 B, corrected energy, collision gate aktif dan
`output_profile=summary`. Tidak ada interval 15 s.

Legacy output goes to `artifacts/experiments/apc_load_sweep/full_factorial_v1/`.
Namespace ini terpisah dari hasil lama; file historis dan analisis sebelumnya
tidak dibaca atau ditimpa. Jika direktori tujuan sudah terisi, periksa dulu
manifest dan fingerprint sumber; jangan menghapusnya untuk mengatasi konflik.

### Historical Summary-vs-Detailed acceptance pilot (manual)

Pilot menjalankan 12 run validasi: tiga skenario (50/90, 150/60, 200/30),
FF/AA, masing-masing `summary` dan `detailed`, seed 30. Perintah ini **benar-benar
menjalankan simulasi**; jangan jalankan sebelum siap. Hasil kesetaraan numerik,
runtime, ukuran/file output dan peak RSS ditulis ke direktori pilot tersendiri.

```bash
cd "$HOME/Documents/LoRaPhyMAC"
MPLCONFIGDIR=/tmp/loraphymac-mpl "$HOME/Documents/LoRaPhyMAC/.venv-output-profile/bin/python" \
  -m loraphymac.experiments.apc_load_sweep.profile_acceptance_pilot --run
```

Validator membandingkan hash tabel numerik summary dan detailed secara eksak,
serta offered schedule, konfigurasi fisik, topology dan routing. Hasil pilot
belum dapat diasumsikan mewakili beban 200 node sebelum pilot ini dijalankan.

## Dry-run tanpa simulasi

```bash
cd "$HOME/Documents/LoRaPhyMAC"
MPLCONFIGDIR=/tmp/loraphymac-mpl "$HOME/Documents/LoRaPhyMAC/.venv-output-profile/bin/python" \
  -m loraphymac.experiments.apc_load_sweep.run \
  src/loraphymac/experiments/apc_load_sweep/configs/ff_full_factorial_v1.json --dry-run
MPLCONFIGDIR=/tmp/loraphymac-mpl "$HOME/Documents/LoRaPhyMAC/.venv-output-profile/bin/python" \
  -m loraphymac.experiments.apc_load_sweep.run \
  src/loraphymac/experiments/apc_load_sweep/configs/aa_full_factorial_v1.json --dry-run
```

Kedua konfigurasi masing-masing merencanakan 60 run. Periksa `run_count=60`
dan 120 ID unik secara keseluruhan sebelum eksekusi.

## Eksekusi produksi (hanya setelah seluruh gate PASS)

```bash
cd "$HOME/Documents/LoRaPhyMAC"
MPLCONFIGDIR=/tmp/loraphymac-mpl "$HOME/Documents/LoRaPhyMAC/.venv-output-profile/bin/python" \
  -m loraphymac.experiments.apc_load_sweep.run \
  src/loraphymac/experiments/apc_load_sweep/configs/ff_full_factorial_v1.json --fail-fast
MPLCONFIGDIR=/tmp/loraphymac-mpl "$HOME/Documents/LoRaPhyMAC/.venv-output-profile/bin/python" \
  -m loraphymac.experiments.apc_load_sweep.run \
  src/loraphymac/experiments/apc_load_sweep/configs/aa_full_factorial_v1.json --fail-fast
MPLCONFIGDIR=/tmp/loraphymac-mpl "$HOME/Documents/LoRaPhyMAC/.venv-output-profile/bin/python" \
  -m loraphymac.experiments.apc_load_sweep.full_factorial_postprocessing
```

Reporter menolak data yang belum lengkap, duplikat, bukan PASS, tidak berpasangan,
berbeda offered schedule/topology/routing/config fisik, atau mencakup interval
15 s. Ia membaca `all_runs_core_summary.csv` dan tabel ringkas per-run; tidak
memerlukan `event_log.csv`. Goodput didefinisikan sebagai delivered 50-byte
application payload bits dibagi 3600 s. Delay adalah rata-rata paket delivered
(metrik kondisional). Interval kepercayaan dihitung atas seed, bukan paket.

## Konfigurasi terdahulu

`ff_v1.json`, `aa_v1.json`, konfigurasi enam-skenario, hasil `*_r01/`, dan
`analysis_r01/` dipertahankan sebagai provenance historis. Reporter
`postprocessing` lama tetap ditujukan pada kontrak enam-skenario/gabungan lama;
gunakan `full_factorial_postprocessing` hanya untuk grid baru.
