# Data Preprocessing Scripts

These are **one-time offline scripts** used to prepare the data files in `PAPI/`. They are **not** part of the experiment pipeline and do not need to be run unless you want to regenerate the data from scratch.

## Scripts

### `preprocess_dataset.py`

Clusters the raw IPIP-NEO-120 survey responses into 300 representative subjects using KMeans, and creates the train/test question split.

**Input:** `PAPI/IPIP_NEO_120.csv` (download from [Google Drive](https://drive.google.com/file/d/1KRhpTCwSMS47GYnmHwYRPnmxF6FOGYTf/view?usp=sharing))

**Output (already in `PAPI/`):**
- `selected_IPIP300_samples.json` — 300 clustered subject profiles
- `mpi_300_split.json` — train (120) / test (180) question index split

### `get_personality_prompt.py`

Generates natural-language personality descriptions for each subject by feeding their training item responses through Llama-3-70B-Instruct. Used to produce the prompts for the P² (Personality Prompt) baseline.

**Requires:** Llama-3-70B-Instruct (4-bit quantized, ~35GB VRAM)

**Output (already in `PAPI/`):**
- `personality_prompt.json` — per-subject personality descriptions
