# Standar Penamaan Direktori Runner Eksperimen

Format direktori runner:

```text
run_YYMMDD_experimentname_rNN
```

Contoh: `run_261112_modelsweep_r01`.

- `run` adalah awalan tetap.
- `YYMMDD` adalah tanggal registrasi atau eksekusi runner dalam zona waktu WIB.
- `experimentname` memakai `lowercase_snake_case`.
- `rNN` adalah nomor revisi berurutan per eksperimen; eksekusi pertama memakai `r01`.
- ID direktori runner berbeda dari ID setiap simulasi individual.
- Direktori runner yang sudah selesai tidak digunakan ulang atau ditimpa.
- Waktu simulasi asli tetap dicatat pada provenance eksperimen; tanggal direktori tidak mengubah atau menggantikan waktu tersebut.
- Runner yang selesai dan tervalidasi diperlakukan sebagai immutable.

## Penerapan pertama

`run_261112_modelsweep_r01` menstandarkan lokasi hasil MAIN_W4 yang telah dijalankan sebelumnya. `261112` adalah tanggal registrasi/relokasi dalam WIB, bukan tanggal simulasi. Identitas eksperimen `main_w4_fixed_reference`, run ID, konfigurasi, waktu eksekusi, dan hasil numeriknya tetap mengikuti provenance asli.

Konvensi ini diterapkan pada APC_COMPARE_MODEL_SWEEP dalam pekerjaan ini; direktori eksperimen lain tidak diubah.
