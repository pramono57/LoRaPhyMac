# Audit deprecasi implementasi legacy R1.7

Tanggal audit: 11 November 2026. Audit dependency dilakukan secara read-only. Tidak ada source, config, notebook, atau artefak historis yang diubah atau dihapus. APC Load Sweep production 120-run (FF 60/60, AA 60/60 PASS) menjadi implementasi R1.7 yang berlaku.

## Lineage dan perbedaan kontrak

R1.7 awal dikembangkan di `scalability_relay_bottleneck` sebagai beban/load dan relay bottleneck sensitivity. Pada keputusan penelitian **1 November 2026**, `apc_load_sweep` ditetapkan sebagai eksperimen R1.7 yang berlaku. Grid aktif dikembangkan menjadi 4 node counts × 3 reporting intervals × 5 seeds × 2 mode (FF/AA) = **120 runs**. Grid lama R1.7 Load memakai 3 node counts × 3 intervals × 5 seeds × 2 mode = **90 runs**. Karena kontrak dan tujuan eksperimennya berubah, hasil dua grid tidak boleh dicampur.

| Atribut | R1.7 Load legacy | R1.7 aktif: APC Load Sweep |
|---|---|---|
| Profile/contract | `scalability_r1_7_load_v1` / `scalability_r1_7_v1` | `apc_load_sweep_full_factorial_v1` |
| Node counts | 50, 100, 200 | 50, 100, 150, 200 |
| Reporting intervals | 15, 30, 60 s | 30, 60, 90 s |
| Seeds | 30, 36, 42, 48, 56 | 30, 36, 42, 48, 56 |
| Model | FF, AA | FF, AA |
| Run count | 45/model; 90 total | 60/model; 120 total |
| Latest evidence | Legacy output partial; source fingerprint lama | Latest runner: FF 60/60, AA 60/60 PASS |

`Relay Duty` adalah sensitivitas terpisah (10%/1% configured relay duty, 100 node, 30 s, 5 seed per mode; 20 runs total). APC Load Sweep tidak menyapu relay duty limit tersebut. Karena fungsi ilmiahnya berbeda, Relay Duty **tidak dianggap tergantikan** oleh APC Load Sweep.

## Dependency audit dan keputusan

| File/Directory | Dependency | Current Function | Replacement | Decision |
|---|---|---|---|---|
| `src/loraphymac/experiments/apc_load_sweep/` | R1.7 aktif; runner, kontrak, konfigurasi, postprocessing | Production full-factorial FF/AA 120-run | — | KEEP |
| `src/loraphymac/experiments/scalability_relay_bottleneck/run.py` | Dipakai config pilot dan Relay Duty; CLI umum | Menjalankan multiseed pilot/load/duty melalui pipeline bersama | Tidak diganti; masih punya fungsi aktif | KEEP |
| `src/loraphymac/experiments/scalability_relay_bottleneck/pilot_v1.json` | `test_experiment_layout.py`; paper2 publication config menunjuk output pilot historis | Pilot exploratory 270-run grid | Tidak diganti oleh full-factorial R1.7 | KEEP |
| `src/loraphymac/experiments/scalability_relay_bottleneck/r1_7_contract.py` | Diimpor `contract_registry.py` dan `scientific_output_contract.py`; memuat aturan Relay Duty, legacy Load dan validasi alias APC Load | Validasi config/spec/output lintas tiga profile | Pindahkan validasi full-factorial ke `apc_load_sweep/contract.py`; pertahankan/extract Relay Duty contract; pertahankan pembacaan profile lama selama backward compatibility diperlukan | MIGRATE, bukan DELETE langsung |
| `configs/r1_7_load_ff_v1.json`, `configs/r1_7_load_aa_v1.json` | Dibaca oleh legacy contract tests dan dokumentasi; output legacy | Grid 90-run R1.7 terdahulu | Config `apc_load_sweep/configs/{ff,aa}_full_factorial_v1.json` untuk grid resmi 120-run | DEPRECATE; DELETE CANDIDATE setelah tes/reader legacy diselesaikan |
| `configs/r1_7_relay_duty_ff_v1.json`, `configs/r1_7_relay_duty_aa_v1.json` | Relay Duty profile, `r1_7_contract.py`, output/reporting | Menguji efek batas duty relay; bukan grid load APC | Tidak ada pengganti teridentifikasi | KEEP |
| `tests/unit/test_scalability_r1_7_contract.py` | Membaca empat config lama; mengunci grid load dan duty | Uji spesifikasi lama | Pisahkan tes legacy-load (backward compatibility/deprecation) dari tes Relay Duty | MIGRATE; jangan hapus sebelum keputusan kompatibilitas |
| `tests/unit/test_scientific_output_contract.py` | Menguji registry profiles `R1_7_LOAD_PROFILE`, `R1_7_DUTY_PROFILE`, serta APC | Menjamin output contract valid | Kontrak APC aktif sendiri; Relay Duty tetap diregistrasikan; legacy Load dapat ditandai deprecated | KEEP lalu MIGRATE bersama registry |
| `src/loraphymac/experiments/common/contract_registry.py` | Impor semua fungsi dari `r1_7_contract.py`; mendaftarkan APC legacy, R1.7 Load lama, Relay Duty, dan APC full-factorial | Registry contract validation yang masih dipakai runtime | Alihkan APC full-factorial sepenuhnya ke APC module; pertahankan Relay Duty; deprecate legacy Load tanpa merusak config lama | MIGRATE, perubahan berisiko lintas eksperimen |
| `src/loraphymac/experiments/common/scientific_output_contract.py` | Mendaftarkan collection/persistence profiles bagi APC dan R1.7 legacy/duty | Preflight kebutuhan scientific output | Pertahankan contract duty; buang alias Load hanya setelah kompatibilitas historis dinyatakan tidak diperlukan | MIGRATE |
| `README.md` | Menginstruksikan runner/config lama sebagai R1.7 aktif | Petunjuk eksperimen | Arahkan R1.7 ke `apc_load_sweep`; beri label historical/legacy pada Load lama; pertahankan pilot dan duty | MIGRATE (prioritas dokumentasi) |
| `src/.../scalability_relay_bottleneck/README.md` | Memuat empat command R1.7 Load+Duty lama dan menyebut total 110 run | Petunjuk package/pilot dan sensitivity | Pisahkan dokumentasi pilot, legacy Load, Relay Duty; rujuk APC Load Sweep untuk R1.7 aktif | MIGRATE (prioritas dokumentasi) |
| `pyproject.toml` package-data untuk package scalability | Memasukkan `configs/*.json` dan `notebooks/*.ipynb` untuk wheel | Distribusi config/notebook package, termasuk pilot dan duty | Tetap perlu selama pilot/duty dipertahankan | KEEP |
| `.gitignore` exception notebooks scalability | Menjaga notebook package tetap terlacak meski pola notebook di-ignore | Source distribution | Tetap diperlukan selama notebook aktif dipertahankan | KEEP |
| `tools/run_r17_production.sh` | Memanggil `loraphymac.experiments.apc_load_sweep.run`, kedua config full-factorial dan postprocessing | Launcher produksi R1.7 aktif (bukan launcher legacy) | Tidak perlu diganti fungsinya; nama dapat diperjelas pada migrasi dokumentasi | KEEP |
| `tools/monitor_r17.py` | Membaca output APC Load Sweep 120-run, FF/AA dan lock/log yang sama | Monitor read-only R1.7 aktif | Tidak perlu diganti fungsinya; nama adalah label R1.7 yang berlaku | KEEP |
| `CHANGELOG.md` | Catatan historis perpindahan package Paper2 | Provenance sejarah | Tidak ada | KEEP |
| `docs/OUTPUT_LAYOUT.md` | Menjelaskan layout artifacts scalability dan notebook | Dokumentasi format/output lintas eksperimen | Perbarui hanya bila package output dipisah kelak | KEEP (review kemudian) |
| `src/loraphymac/experiments/scalability_relay_bottleneck/notebooks/scalability_relay_bottleneck_workflow.ipynb` | Alur pilot/scalability exploratory, bukan hanya kontrak Load R1.7 | Workflow eksperimen interaktif | Tidak digantikan APC Load Sweep tanpa pemeriksaan cell pengguna | KEEP |
| `src/loraphymac/publications/paper2_scalability/` dan `configs/publication_v1.json` | Menggunakan output pilot `scalability_relay_bottleneck_v1_pilot_r01`; analisis/figur memakai `relay_duty_cycle` | Workflow Paper2 dan bukti pilot/duty historis | Tidak digantikan oleh R1.7 APC | KEEP |
| `.ipynb_checkpoints/` dan `run-checkpoint.py` di package | Tidak ada import/entrypoint runtime teridentifikasi; konten dapat berupa pekerjaan lokal | Checkpoint editor | Arsip/compare terlebih dahulu | DELETE CANDIDATE hanya setelah pemeriksaan isi dan backup terpisah |
| `artifacts/experiments/scalability_relay_bottleneck/` | Run, manifest, log, output pilot/load/duty | Bukti historis/provenance | Tidak ada | KEEP; jangan ubah/hapus |

Pencarian referensi juga menemukan kontrak active alias `apc_load_sweep_v1` yang masih memakai validator dari `r1_7_contract.py`; alias tersebut berbeda dari full-factorial active profile dan tidak boleh dihapus hanya berdasarkan nama. Migrasi harus terlebih dahulu memetakan konfigurasi/artefak yang masih menggunakannya.

## Dependency yang tetap aktif

1. Runner umum dipakai oleh pilot scalability dan Relay Duty.
2. Registry serta scientific output contract masih mengimpor `r1_7_contract.py`.
3. Paper2 publication profile membaca artefak pilot scalability lama dan metrik duty.
4. Package data dan `.gitignore` menjaga config/notebook yang tetap digunakan.
5. `monitor_r17.py` dan `run_r17_production.sh` saat ini menunjuk ke APC Load Sweep official 120-run, jadi keduanya bukan bagian yang digantikan.
6. README package masih menyarankan command legacy load; dokumentasi itu perlu dikoreksi untuk menghindari rerun grid lama yang tidak berlaku.

## Kandidat penghapusan dan struktur yang direkomendasikan

### Kandidat (belum dihapus)

- Empat konfigurasi `r1_7_load_{ff,aa}_v1.json` hanya untuk grid 90-run yang sudah diganti. Setelah menyatakan deprecation dan memisahkan tes/validator, kandidat penghapusan source config.
- Cabang `R1_7_LOAD_PROFILE` serta aturan spesifik load dalam `r1_7_contract.py`, `contract_registry.py`, dan `scientific_output_contract.py`. Hapus hanya jika reader/config historical tidak perlu backward compatibility; jika masih perlu, tandai legacy dan pertahankan reader.
- Checkpoint notebook/script hanya setelah dibandingkan dengan source aktif dan diarsipkan; audit ini tidak menilai isi checkpoint sebagai aman dihapus.

### Jangan hapus

- `run.py`, `pilot_v1.json`, notebook workflow, package, Relay Duty configs/contract/tests, Paper2 publication code/data, production R1.7 tools, dan seluruh artifacts historis. Semuanya memiliki dependency/fungsi terpisah dari APC Load Sweep.

### Struktur akhir yang disarankan

```text
experiments/
  apc_load_sweep/                 # R1.7 aktif: 120-run FF/AA
  scalability_relay_bottleneck/  # pilot dan Relay Duty; tanpa klaim R1.7 Load aktif
    contracts/relay_duty.py       # target migrasi bila ekstraksi disetujui
    configs/pilot_v1.json
    configs/relay_duty_{ff,aa}_v1.json
    notebooks/...
```

Sebelum implementasi cleanup, urutan aman: (1) koreksi README, (2) tandai konfigurasi Load lama deprecated, (3) migrasikan validasi APC full-factorial sepenuhnya ke `apc_load_sweep/contract.py`, (4) ekstrak/pertahankan Relay Duty contract dan registry, (5) ubah/pisahkan tes, (6) baru evaluasi penghapusan kandidat config lama. Tidak ada langkah tersebut dilakukan dalam audit ini; memerlukan persetujuan cleanup terpisah.

## Kesimpulan

Implementasi Load R1.7 lama dapat dipensiunkan, tetapi package `scalability_relay_bottleneck` **tidak aman dihapus secara keseluruhan**. Relay Duty dan pilot memiliki fungsi serta consumer independen. Prioritas teraman adalah memperbaiki dokumentasi dan mengisolasi kontrak legacy; hapus kandidat hanya setelah semua consumer historis/runtime telah ditangani secara eksplisit.
