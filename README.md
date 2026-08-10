# Source-Modality Monitoring in Vision-Language Models

This repository contains the code for running experiments presented in [Source-Modality Monitoring in Vision-Language Models](https://arxiv.org/abs/2604.22038), published as a conference paper at COLM 2026.

## Abstract

> We define and investigate *source-modality monitoring* – the ability of multimodal models to track and communicate the input source from which pieces of information originate. We consider source-modality monitoring as an instance of the more general *binding problem*, and evaluate the extent to which models exploit syntactic versus semantic signals in order to bind words like *image* in a user-provided prompt to specific components of their input and context (i.e., actual images). Across experiments spanning 11 vision-language models (VLMs) performing target-modality information retrieval tasks, we find that both syntactic and semantic signals play an important role, but that the latter tends to outweigh the former in cases when modalities are highly distinct distributionally. We discuss the implications of these findings for model robustness, and in the context of increasingly multimodal agentic systems.

## Table of Contents

- [Source-Modality Monitoring in Vision-Language Models](#source-modality-monitoring-in-vision-language-models)
  - [Abstract](#abstract)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
  - [Overall Organization](#overall-organization)
  - [Behavioral Evaluation](#behavioral-evaluation)
  - [LLM Judge Scoring](#llm-judge-scoring)
  - [Symbolic Binding with Arbitrary Labels](#symbolic-binding-with-arbitrary-labels)
  - [Representation Analysis](#representation-analysis)
  - [Marker Perturbations](#marker-perturbations)
  - [Freeze-Remove Intervention](#freeze-remove-intervention)
  - [Learned Transformation Vectors](#learned-transformation-vectors)
  - [Running at Scale with Slurm](#running-at-scale-with-slurm)
  - [Aggregated Results](#aggregated-results)
  - [Figures and Tables](#figures-and-tables)
  - [How to Cite](#how-to-cite)
  - [License](#license)

## Installation

Use the command below to set up the environment:

```
pip install -r requirements.txt
```

The LLM judge calls the OpenAI API. Copy the example environment file and fill in your key:

```
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=          # optional, used with the --openrouter flag
```

## Dataset Setup

We use two image-captioning datasets.

**Flickr30k** is downloaded automatically from HuggingFace (`lmms-lab/flickr30k`, `test` split). No setup required.

**MSCOCO 2017** must be downloaded separately from [cocodataset.org](https://cocodataset.org/#download). You need `train2017`, `val2017`, and the caption annotations. Point the code at the directory containing them:

```
export COCO_ROOT=/path/to/coco
```

The expected layout is:

```
$COCO_ROOT/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

Both datasets are permuted with a fixed seed and subsampled to 4,000 train / 2,000 validation / 2,000 test examples. Inconsistent image-caption pairs are constructed by sampling a caption whose sentence-embedding cosine similarity to *any* of the image's five gold captions falls below `0.2`, computed with `sentence-transformers/all-mpnet-base-v2`. See `--inconsistent_sim_threshold` to change this.

## Overall Organization

The main scripts are located under the `./src` folder. Each script is run directly with command-line arguments:

```
cd src
python prompt_evaluation.py --seed 42 --work_dir .. --model_name ...
```

Note that scripts are run **from inside `./src`** (they import `data` and `utils` as top-level packages), while `--work_dir` points at the repository root, which is where `results/` will be written. Every script documents its full argument list under `--help`.

Experiments come in two stages. Generation scripts run the VLM and save raw per-example predictions under `results/`; the judge scripts then score those predictions with an LLM judge and write to `results_gpt_eval/`. The analysis notebooks read the judged files.

```
src/           experiment code
scripts/       Slurm launchers that sweep the full parameter grids
notebooks/     figure and table generation
summaries/     aggregated per-condition CSVs (included)
tools/         aggregate_results.py, which produces summaries/
```

## Behavioral Evaluation

This is the core target-modality retrieval task. To run it on a single configuration:

```
cd src
python prompt_evaluation.py \
  --seed 42 \
  --work_dir .. \
  --model_name Qwen/Qwen2.5-VL-32B-Instruct \
  --dataset mscoco \
  --split test \
  --version 1 \
  --prompt_format image_caption \
  --input_type inconsistent \
  --modality_to_report image \
  --order icq \
  --modify_inputs none \
  --batch_size 8
```

Some of the important arguments include:

- `model_name`: the HuggingFace model string for the model to be evaluated. The 11 models in the paper are:
  - `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct`
  - `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`
  - `OpenGVLab/InternVL3-8B-hf`, `OpenGVLab/InternVL3-14B-hf`
  - `llava-hf/llava-1.5-7b-hf`, `llava-hf/llava-onevision-qwen2-7b-ov-hf`
  - `Salesforce/instructblip-vicuna-7b`
- `dataset`: the dataset to be used for evaluation, choose from `mscoco` and `flickr30k`
- `input_type`: defines how the caption and image relate to each other, choose from
  - `inconsistent` (the caption and image describe different scenes — the main condition),
  - `consistent` (the caption genuinely describes the image),
  - `image_only` (image with no caption — the single-modality baseline),
  - `text_only` (caption with no image — the single-modality baseline)
- `modality_to_report`: defines the target modality, choose from `image` and `text`
- `prompt_format`: the word used to refer to the text source, choose from `image_caption` ("Caption: X."), `image_text` ("Text: X."), and `image_document` ("Document: X."). This is the manipulation behind Figure 5.
- `order`: the order in which the two sources appear, choose from `icq` (image, then caption, then query) and `ciq` (caption first). Every configuration in the paper is run both ways and averaged, to rule out input-order effects. InstructBLIP is the exception, as it has a fixed input order.
- `modify_inputs`: the marker perturbation, see [Marker Perturbations](#marker-perturbations) below. Use `none` for the unperturbed condition.
- `version`: the run version. Use `1` for the first version.

Results are written to a path encoding the full configuration:

```
{work_dir}/results/behavioral_evaluation/modification_{modify_inputs}/{model}/{dataset}/{input_type}/{modality_to_report}/{prompt_format}_{order}_s{seed}.json
```

Sweeping this across all 11 models produces the aggregated selectivity scores in Figure 2.

## LLM Judge Scoring

Model outputs are free-form text, so they cannot be scored by exact string matching. We use `GPT-5.4-mini` as an LLM judge to decide whether each response is grounded in the image, the caption, or neither. Judge-human agreement is 88% (Appendix D).

To score a raw result file:

```
cd src
python gpt_judge_evaluation.py \
  --work_dir .. \
  --input_json ../results/behavioral_evaluation/modification_none/Qwen/Qwen2.5-VL-32B-Instruct/mscoco/inconsistent/image/image_caption_icq_s42.json
```

Some of the important arguments include:

- `input_json`: path to the raw result file produced by `prompt_evaluation.py`. The output path is derived automatically, mirroring the input layout under `results_gpt_eval/`.
- `gpt_model`: the judge model, defaults to `openai/gpt-5.4-mini`
- `openrouter`: route requests through OpenRouter instead of the OpenAI API, using `OPENROUTER_API_KEY`
- `mode`: `async` (default), `sync`, or `batch`. Use `batch` for large jobs at lower cost.
- `concurrency`: number of in-flight requests in async mode, defaults to `32`
- `start` / `end` / `max_instances`: score only a slice of the file, useful for testing before spending on a full run

The judged file records the per-condition totals (`gpt_judge_counts`, `gpt_judgment_hist`) at the top level alongside the per-example judgments, which is what the aggregation step reads.

There are two sibling judges for the other experiment families: `gpt_judge_evaluation_arb.py` for the arbitrary-label runs (it additionally needs `--label_1` and `--label_2` to match the source directory), and `gpt_judge_eval_transformation_vec.py` for the intervention runs.

## Symbolic Binding with Arbitrary Labels

To test whether VLMs can use symbols as pure indexing devices, we replace the modality words with content-free labels, so that success requires binding each content span to its assigned label with no semantic cue to fall back on.

```
cd src
python prompt_evaluation_arbitrary_labels.py \
  --seed 42 \
  --work_dir .. \
  --model_name Qwen/Qwen2.5-VL-32B-Instruct \
  --dataset mscoco \
  --split test \
  --version 1 \
  --input_type inconsistent \
  --modality_to_report image \
  --order icq \
  --label_1 Dax \
  --label_2 Wug
```

- `label_1`: the arbitrary label bound to the **image** content
- `label_2`: the arbitrary label bound to the **caption** content

The paper uses the pairs `Alpha`/`Beta` and `Dax`/`Wug`, each in both assignment directions (so `Dax`/`Wug` and `Wug`/`Dax`), to counterbalance any preference for a particular label string. This produces Figure 3.

## Representation Analysis

This measures whether image and text tokens are already distinguishable from their distributional properties alone, before any contextual processing by the language model. We take token representations at the embedding layer, compute within- and cross-modality cosine similarities, and fit a linear probe to classify token modality with 3-fold cross-validation, alongside a shuffled-label control.

```
cd src
python representation_analysis.py \
  --work_dir .. \
  --model_name Qwen/Qwen2.5-VL-32B-Instruct \
  --dataset mscoco \
  --span_type content \
  --num_samples 200 \
  --layer_idx 0 \
  --seed 42
```

- `span_type`: which token positions to analyze, choose from `content` (the image and caption content tokens), `start`, and `end` (the marker positions)
- `num_samples`: number of instances sampled per dataset, defaults to `200`
- `layer_idx`: which layer to read representations from. `0` is the embedding layer, as used in the paper.
- `control_shuffle_seed`: seed for the shuffled-label control probe

To collect the results across all model-dataset pairs into the LaTeX table in Appendix H:

```
python summarize_representation_analysis_table.py --results-dir ../results/representation_analysis
```

This produces Table 6.

## Marker Perturbations

To separate symbolic from distributional signals, we structurally perturb the modality marker tokens. The three conditions in Table 1 are set through `--modify_inputs` on `prompt_evaluation.py`:

- `none`: unperturbed, the original modality markers are preserved
- `remove`: both the image and caption markers are deleted
- `swap`: the image and caption markers are exchanged

```
cd src
for cond in none remove swap; do
  python prompt_evaluation.py \
    --seed 42 --work_dir .. --version 1 \
    --model_name Qwen/Qwen2.5-VL-32B-Instruct \
    --dataset mscoco --split test \
    --prompt_format image_caption \
    --input_type inconsistent \
    --modality_to_report image \
    --order icq \
    --modify_inputs $cond
done
```

If binding were purely symbolic, `remove` should drop selectivity to chance and `swap` should reverse it. Neither happens, which is the central evidence that distributional signals carry modality identity too. This produces Figure 4.

Repeating the sweep across `--prompt_format image_caption image_text image_document` produces Figure 5, showing that models lean on markers heavily when the text is called a *document* and barely at all when it is called *text*.

Additional values of `--modify_inputs` (`i2c`, `c2i`, `swap_start`, `text_substrate`) are supported by the code but are not used in the paper.

## Freeze-Remove Intervention

This tests whether marker information propagates into downstream content-token representations during contextualization. We collect hidden activations at the content-token positions from a clean run with markers intact, then patch those activations into a second run in which the markers are removed, and let computation continue from there.

```
cd src
python freeze_content_remove.py \
  --seed 42 \
  --work_dir .. \
  --model_name Qwen/Qwen2.5-VL-32B-Instruct \
  --dataset mscoco \
  --split test \
  --version 1 \
  --prompt_format image_caption \
  --input_type inconsistent \
  --modality_to_report image \
  --order icq
```

Results are written under `results/behavioral_evaluation/modification_freeze_content_remove/`, matching the layout of the other behavioral runs so the same judge and aggregation steps apply.

Selectivity is partially restored relative to plain marker removal, indicating that modality identity has been written into the content tokens rather than staying localized at the marker positions. This produces Figure 6; the intervention itself is illustrated in Figure 11.

## Learned Transformation Vectors

Here we ask whether the marker and content representations can be manipulated to *induce* source misattribution. We learn two vectors, δ₁ and δ₂, added to the image span and caption span respectively at a given layer, optimized so the model reports the non-queried modality. The model's own weights stay frozen; only the two vectors are trained.

**Step 1 — train the vectors:**

```
cd src
python train_transformation_vec.py \
  --model_name Qwen/Qwen2.5-VL-32B-Instruct \
  --dataset mscoco \
  --work_dir .. \
  --seed 42 \
  --span_type marker \
  --layer_depth 0.0
```

- `span_type`: where the vectors are added, choose from
  - `marker` (at the symbolic marker token positions),
  - `content` (at the modality-specific content token positions),
  - `baseline_first` and `baseline_last` (at the first / last token position — the prefix-tuning and function-vector style baselines from Appendix L.2)
- `layer_depth`: the *relative* depth at which the vectors are added, from `0.0` (first layer) to `1.0` (last). The paper sweeps `0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0`.
- `num_delta_samples`: training examples drawn from the MSCOCO train split, defaults to `4000`
- `lr`, `n_epochs`, `grad_accum_steps`: optimization settings, defaulting to `1e-2`, `1`, and `1`
- `train_orders`: input orders to train over, defaults to both `icq` and `ciq`

Trained vectors are saved as `deltas.pt` under `results/train_transformation_vec/`.

**Step 2 — evaluate them:**

```
python eval_transformation_vec.py \
  --model_name Qwen/Qwen2.5-VL-32B-Instruct \
  --dataset mscoco \
  --work_dir .. \
  --train_seed 42 \
  --intervention marker \
  --layer_depth 0.0
```

- `intervention`: must match the `span_type` used at training time
- `train_seed`: identifies which trained checkpoint to load. Combined with `--intervention` and `--layer_depth` this resolves the `deltas.pt` path automatically, or pass `--deltas_path` explicitly.
- `num_eval_samples`: held-out validation examples, defaults to `100`
- `split` / `mscoco_split`: which split to evaluate on, defaulting to `val`

The paper repeats the whole procedure across **3 random seeds** and reports the mean with standard deviation. Evaluation saves both baseline and intervention predictions for each target modality, which the judge then scores. This produces Figures 8, 14, 15, and 16.

## Running at Scale with Slurm

Reproducing a full figure means sweeping many configurations, so `./scripts` contains launchers that expand the parameter grids and submit one job each.

| Command | What it sweeps |
|---|---|
| `./scripts/run_behavioral.sh` | 11 models × 2 datasets × 2 orders × 2 target modalities |
| `./scripts/run_arbitrary_labels.sh` | 3 models × 4 label pairs × 2 datasets × 2 orders × 2 modalities |
| `./scripts/run_representation_analysis.sh` | 11 models × 2 datasets |
| `./scripts/run_freeze_remove.sh` | 3 models × 2 datasets × 2 orders × 2 modalities |
| `./scripts/run_transformation_vec.sh train` | 3 models × 4 interventions × 9 depths × 3 seeds |
| `./scripts/run_transformation_vec.sh eval` | the same grid, evaluating the trained vectors |
| `./scripts/run_gpt_judge.sh` | judges everything currently under `results/` |

Cluster settings live in `scripts/config.sh` and are all overridable from the environment:

```
DRY_RUN=1 ./scripts/run_behavioral.sh                        # print jobs without submitting
SUBMIT=bash ./scripts/run_behavioral.sh                      # run serially, no Slurm
PARTITION=gpu-he TIME=10:00:00 ./scripts/run_behavioral.sh   # override resources
CONDITIONS="none remove swap" MODELS_SET=focus ./scripts/run_behavioral.sh
PROMPT_FORMATS="image_caption image_text image_document" ./scripts/run_behavioral.sh
```

`MODELS_SET=focus` restricts a sweep to the three models used for the mechanistic analyses (Qwen2.5-VL-32B, Gemma-3-12B, InternVL3-14B). GPU counts are assigned per model in `gpus_for()` in `scripts/config.sh`. Always try `DRY_RUN=1` first — the full transformation-vector grid is 324 jobs.

## Aggregated Results

The raw per-example outputs run to several GB and are not included. What ships instead is `summaries/`, containing the per-condition aggregates the figures are drawn from:

| File | Rows | Contents |
|---|---|---|
| `behavioral.csv` | 396 | selectivity and valid-response rate per model × dataset × marker condition |
| `behavioral_arbitrary_labels.csv` | 48 | the Alpha/Beta and Dax/Wug symbolic task |
| `transformation_vec.csv` | 1296 | intervention results per layer depth, span type, and seed |
| `representation_analysis.csv` | 16 | cosine similarities and linear-probe accuracy |

If you have reproduced the raw runs locally, regenerate these with:

```
python tools/aggregate_results.py --raw-root /path/to/dir/containing/results --out summaries/
```

`representation_analysis.csv` reproduces every number in Table 6 of the paper exactly.


## Figures and Tables

**`notebooks/figures_from_summaries.ipynb`** draws the paper's quantitative
figures from the included `summaries/` CSVs alone. It needs no raw data, no GPU
and no model downloads, so it runs immediately after cloning. It **ships with its
cell outputs**, so all the figures are visible on GitHub without running
anything:

| Section | Produces |
|---|---|
| Figure 2 | selectivity and valid-response rates across the 11 VLMs |
| Figure 3 | purely symbolic binding with arbitrary labels |
| Figures 4 and 6 | marker perturbations and the freeze-remove intervention |
| Figure 5 | image-caption vs image-text vs image-document (also Figures 12, 13) |
| Figures 8, 14, 15 | learned transformation vectors across layer depth |
| Figure 16 | first- and last-token position baselines (Appendix L.2) |
| Table 6 | distributional separation of image and text tokens |

## How to Cite

```
@inproceedings{hua2026source,
  title     = {Source-Modality Monitoring in Vision-Language Models},
  author    = {Etha Tianze Hua and Tian Yun and Ellie Pavlick},
  booktitle = {Third Conference on Language Modeling},
  year      = {2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
