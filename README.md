# polyA2

Pipeline for calling allelic variants and polyA tract lengths from single-cell
LUSTRE data.

## Installation

### With conda (recommended)

`pysam` and `h5py` are easiest to install from the bioconda/conda-forge channels,
which is what `environment.yml` uses.

```bash
git clone <this-repo> polyA2
cd polyA2
conda env create -f environment.yml
conda activate polyA2
```

### With pip

```bash
git clone <this-repo> polyA2
cd polyA2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer is required.

### One-time `hist_map.pickle` conversion

If you pulled an older snapshot of the repo, `hist_map.pickle` may still be in
its legacy float128 format and will fail to load with:

```
TypeError: data type 'f16' not understood
```

Convert it once (works on any platform):

```bash
python tools/convert_hist_map.py
```

The file in the current repo is already converted, so this is only needed for
historical checkouts.

## Running the pipeline

```bash
python main_script.py \
    --input reads.bam \
    --insertions insertions.csv \
    --samplesheet samplesheet.csv \
    --prefix run1 \
    --threads 8
```

Each stage writes a pickle (e.g. `run1_res_dict.pkl`, `run1_histogram_dfs.pkl`,
`run1_phase_allele_snv.pkl`, …) plus a final `run1_insertion_summary.csv`. If a
stage's output already exists, the script loads it and skips the work; pass
`--overwrite` to force re-computation.

## Tests

```bash
python -m pytest tests/
```
