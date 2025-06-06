from pathlib import Path

# change to path where the Argoverse2 Sensor dataset is downloaded
AV2_DATA_DIR = Path('E:/ArgoVerse2')
TRACKER_DOWNLOAD_DIR = Path('E:/refav/RefAV/tracker_downloads')
SM_DOWNLOAD_DIR = Path('E:/refav/RefAV/av2_sm_downloads')

# path to cached atomic function outputs, likely does not exist for you
CACHE_PATH = Path('E:/refav/RefAV/refav-cache')

#input directories, do not change
EXPERIMENTS = Path('E:/AV2_code/RefAV/run/experiments.yml')
REFAV_CONTEXT = Path('E:/AV2_code/RefAV/refAV/llm_prompting/refAV_context.txt')
AV2_CATEGORIES = Path('E:/AV2_code/RefAV/refAV/llm_prompting/av2_categories.txt')
PREDICTION_EXAMPLES = Path('E:/AV2_code/RefAV/refAV/llm_prompting/prediction_examples.txt')

#output directories, do not change
SM_DATA_DIR = Path('E:/refav/RefAV/run/output/sm_dataset')
SM_PRED_DIR = Path('E:/AV2_FUCK_YOU_DATA/sm_predictions')
LLM_PRED_DIR = Path('E:/AV2_FUCK_YOU_DATA/llm_code_predictions')
TRACKER_PRED_DIR = Path('E:/refav/RefAV/run/output/tracker_predictions')

Vis = Path('E:/AV2_FUCK_YOU_DATA/sm_predictions/exp4')

