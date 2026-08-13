#!/usr/bin/env python
"""Inferencia de BirdNET en un subproceso independiente.

Se ejecuta como script (`if __name__ == "__main__"`) para que el multiprocessing de
BirdNET use `spawn` (por defecto en macOS), evitando los **deadlocks** de `fork`
cuando BirdNET se corre dentro del kernel de Jupyter.

Estrategia robusta, rápida y con memoria acotada:
- Procesa los audios **por lotes** (`--batch-size`): carga cada lote a arrays mono de
  48 kHz (`birdnet_utils.load_mono_48k`), predice y escribe los segmentos de forma
  **incremental** al CSV, y libera la memoria antes del siguiente lote. Así no se
  cargan todos los audios en RAM a la vez (evita OOM con muestras grandes).
- Pre-decodificar a 48 kHz descarta ilegibles/muy cortos y **esquiva el bug de resample**
  de los MP3 corruptos de Xeno-canto (que si no cancela el lote).

Uso:
    python run_birdnet_inference.py --input-csv sample.csv \
        --custom-species-file labels.txt --out-csv segments.csv \
        --n-producers 8 --top-k 5 --batch-size 200

`--input-csv` debe tener columnas: recording_id, species, audio_path.
Salida (`--out-csv`): recording_id, species_true, start_time, end_time, species_name, confidence.
"""
import argparse
import gc
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--custom-species-file", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--n-producers", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--target-sr", type=int, default=48000)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--max-duration-s", type=float, default=180.0,
                    help="trunca audios más largos (acota memoria; 180 s basta para top-1)")
    ap.add_argument("--load-workers", type=int, default=8,
                    help="hilos para decodificar audio en paralelo (aprovecha varios núcleos)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pandas as pd
    import birdnet_utils as bu
    import birdnet

    df_in = pd.read_csv(args.input_csv)
    custom = [l.strip() for l in open(args.custom_species_file, encoding="utf-8") if l.strip()]
    items = [(r.recording_id, r.species, r.audio_path) for r in df_in.itertuples()
             if isinstance(getattr(r, "audio_path", None), str) and os.path.exists(r.audio_path)]
    print(f"[infer] grabaciones con audio en disco: {len(items)}/{len(df_in)} "
          f"| lotes de {args.batch_size} | especies permitidas: {len(custom)}", flush=True)

    cols = ["recording_id", "species_true", "start_time", "end_time", "species_name", "confidence"]
    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)

    model = birdnet.load("acoustic", "2.4", "tf")
    t0 = time.time()
    first, done, total_recs, failed_batches = True, 0, 0, 0

    from concurrent.futures import ThreadPoolExecutor

    def _load_one(item):
        rec, sp, path = item
        a = bu.load_mono_48k(path, target_sr=args.target_sr)
        if a is None:
            return None
        arr, sr = a
        max_n = int(args.max_duration_s * sr)
        if len(arr) > max_n:                    # truncar audios muy largos
            arr = arr[:max_n]
        return (arr, sr, rec, sp)

    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        # Decodificar/resamplear en paralelo (libsndfile y scipy liberan el GIL)
        with ThreadPoolExecutor(max_workers=args.load_workers) as ex:
            loaded = [r for r in ex.map(_load_one, chunk) if r is not None]
        arrs = [(arr, sr) for arr, sr, _, _ in loaded]
        meta = [(rec, sp) for _, _, rec, sp in loaded]
        if arrs:
            try:
                res = model.predict_arrays(
                    arrs, top_k=args.top_k, n_producers=args.n_producers,
                    default_confidence_threshold=0.0, custom_species_list=custom,
                    show_stats=None)
                seg = res.to_dataframe()
                idx = seg["input"].astype(int).values
                seg["recording_id"] = [meta[j][0] for j in idx]
                seg["species_true"] = [meta[j][1] for j in idx]
                seg[cols].to_csv(args.out_csv, mode="w" if first else "a",
                                 header=first, index=False)
                first = False
                total_recs += seg["recording_id"].nunique()
                del res, seg
            except Exception as e:
                failed_batches += 1
                print(f"[infer]   lote {i//args.batch_size+1} FALLÓ, se omite: {str(e)[:80]}", flush=True)
        done += len(chunk)
        del arrs, meta
        gc.collect()
        rate = done / max(time.time() - t0, 1e-9)
        print(f"[infer] {done}/{len(items)} audios | ~{total_recs} grabaciones | "
              f"{rate:.1f} audios/s | {(time.time()-t0)/60:.1f} min", flush=True)

    if first:  # nunca se escribió nada
        pd.DataFrame(columns=cols).to_csv(args.out_csv, index=False)
    print(f"[infer] LISTO en {(time.time()-t0)/60:.1f} min | grabaciones con predicción: {total_recs} "
          f"| lotes fallidos: {failed_batches}", flush=True)


if __name__ == "__main__":
    main()
