import argparse
from pathlib import Path
import os
import yaml
import multiprocessing
import refAV.paths as paths
from refAV.dataset_conversion import (
    separate_scenario_mining_annotations,
    pickle_to_feather,
    create_gt_mining_pkls_parallel
)
from refAV.parallel_scenario_prediction import run_parallel_eval
from refAV.eval import evaluate_pkls, combine_pkls, combine_pkls_pred, combine_pkls_stream

# import os
# PROXY = "http://127.0.0.1:7890"
# for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
#     os.environ[k] = PROXY
def main():
    parser = argparse.ArgumentParser(description="Run full RefAV pipeline for a given experiment")
    parser.add_argument(
        "--exp_name", type=str, required=True,
        help="Name of the experiment from experiments.yml to run."
    )
    parser.add_argument(
        "--procs_per_task", type=int, default=3,
        help="Base number of processes for each eval.py task. Extra CPUs will be distributed."
    )
    args = parser.parse_args()

    # Load experiment config
    with open(paths.EXPERIMENTS, 'rb') as file:
        config = yaml.safe_load(file)

    exp_cfg = config.get(args.exp_name)
    if exp_cfg is None:
        raise ValueError(f"Experiment '{args.exp_name}' not found in {paths.EXPERIMENTS}")

    exp_name = exp_cfg['name']
    llm = exp_cfg['LLM']
    tracker = exp_cfg['tracker']
    split = exp_cfg['split']

    # Validate config values
    if llm not in config.get("LLM", []):
        print('Experiment uses an invalid LLM')
    if tracker not in config.get("tracker", []):
        print('Experiment uses invalid tracking results')
    if split not in ['train', 'test', 'val']:
        print('Experiment must use split train, test, or val')


    # Scenario mining GT
    if split in ['val', 'train']:
        sm_feather = paths.SM_DOWNLOAD_DIR / f'scenario_mining_{split}_annotations.feather'
        sm_data_split_path = paths.SM_DATA_DIR / split
        if not sm_data_split_path.exists():
            separate_scenario_mining_annotations(sm_feather, sm_data_split_path)
            create_gt_mining_pkls_parallel(
                sm_feather,
                sm_data_split_path,
                num_processes=max(1, int(.9 * os.cpu_count()))
            )

    # Tracker predictions conversion
    tracker_predictions_pkl = Path(f'E:/refav/RefAV/tracker_downloads/{tracker}_{split}.pkl')
    tracker_predictions_dest = paths.TRACKER_PRED_DIR / tracker / split
    if not tracker_predictions_dest.exists():
        av2_data_split = paths.AV2_DATA_DIR / split
        pickle_to_feather(
            av2_data_split,
            tracker_predictions_pkl,
            tracker_predictions_dest
        )

    # Parallel scenario prediction
    log_prompts_path = paths.SM_DOWNLOAD_DIR / f'log_prompt_pairs_{split}.json'
    run_parallel_eval(
        exp_name,
        log_prompts_path,
        args.procs_per_task
    )

    # Combine and evaluate
    experiment_dir = paths.SM_PRED_DIR / exp_name
    combined_preds = combine_pkls_pred(experiment_dir, log_prompts_path)

    if split in ['val', 'train']:
        combined_gt = combine_pkls(paths.SM_DATA_DIR, log_prompts_path)
    else:
        combined_gt = Path(
            f'/home/crdavids/Trinity-Sync/av2-api/output/eval/{split}/latest/combined_gt_{split}.pkl'
        )

    metrics = evaluate_pkls(combined_preds, combined_gt, experiment_dir)
    print(metrics)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
