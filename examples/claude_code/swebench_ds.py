from datasets import load_dataset

src = "/gfs/platform/public/infra/wyb/SWE-bench"
dst = "/gfs/platform/public/infra/wyb/SWE-bench_hf_disk"

for split in ["train", "dev", "test"]:
    ds = load_dataset(
        "parquet",
        data_files={split: f"{src}/data/{split}-*.parquet"},
        split=split,
    )
    ds.save_to_disk(f"{dst}/{split}")