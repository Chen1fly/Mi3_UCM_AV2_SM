## RefAV: Mining Referred Scenarios in Autonomous Vehicle Datasets using LLMs

<p align="center">
  <img src="docs/figures/pipeline.png" alt="RefAV Method">
</p>

A single autonomous vehicle will stream about ~4TB of data per hour with a full stack of camera and lidar sensors. The vast majority of this data comes from uninteresting scenarios -- the ego vehicle driving straight down a lane, possibly with another car in front of it. It can be prohibitively expensive to retrive and label specific scenarios for ego-behaivor evaluation, safety testing, or active learning at scale.

RefAV serves as the baseline for the 2025 Argoverse2 Scenario Mining Challenge. It utilizes an LLM to construct composable function calls from a set of hand-crafted atomic functions such as "turning" or "has_objects_in_front". Given a prompt, the LLM outputs a composable function that narrows down a set of bounding box track predictions to the set that best corresponds to the prompt. This method can be thought as equivalent to querying a SQL database.  

:rotating_light: Top performing teams can win cash prizes! :rotating_light:

:1st_place_medal: 1st Place: $5,000

:2nd_place_medal: 2nd Place: $3,000

:3rd_place_medal: 3rd Place: $2,000

To be eligible for prizes, teams must submit a technical report, open source their code, and provide instructions on how to reproduce their results. 

The scenario mining test split and EvalAI leaderboard will both open on May 7th, 2025. The scenario mining train and val split are available for download now. You may test your method using the available val leaderboard. The val split results are not factored into the competition. 

### Installation

Using [Conda](https://anaconda.org/anaconda/conda) is recommended for environment management
```
conda create -n refav python=3.10
conda activate refav
```

All of the required libaries and packages can be installed with

```
pip install -r requirements.txt
```

Running this code requires downloading the Argoverse2 test and val splits. Run the commands below to download the entire sensor dataset.
More information can be found in the [Argoverse User Guide](https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data).
```
conda install s5cmd -c conda-forge

export DATASET_NAME="sensor"  # sensor, lidar, motion_forecasting or tbv.
export TARGET_DIR="$HOME/data/datasets"  # Target directory on your machine.

s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" $TARGET_DIR
```
It also requies downloading the scenario-mining add on. 
```
export TARGET_DIR="$(pwd)/av2_sm_downloads"
s5cmd --no-sign-request cp "s3://argoverse/tasks/scenario_mining/*" $TARGET_DIR
```

### Generating Detections and Tracks
See the [LT3D repository](https://github.com/neeharperi/LT3D) for information on training a baseline detector and tracker on the Argoverse 2 dataset. The tutorial notebook includes code to download a sample set of tracks.

### Running the Code

All of the code necessary for unpacking the dataset, generating referred track predictions,
and evaluating the predictions against the ground truth can be found in the `tutorial.ipynb` file.
It also includes some basic tutorials about how to define and visualize a scenario.

### Benchmark Evaluation

| **Metric** | **Description** |
|------------|-----------------|
| HOTA-Temporal | HOTA on temporally localized tracks. |
| HOTA | HOTA on the full length of a track |
| Timestamp F1 | Timestamp level classification metric |
| Scenario F1 | Scenario level classification metric. |

### Submission Format

The evaluation expects a dictionary of lists of dictionaries
```python
{
      <(log_id,prompt)>: [
            {
                  "timestamp_ns": <timestamp_ns>,
                  "track_id": <track_id>
                  "score": <score>,
                  "label": <label>,
                  "name": <name>,
                  "translation_m": <translation_m>,
                  "size": <size>,
                  "yaw": <yaw>,
            }
      ]
}
```

log_id: Log id associated with the track, also called seq_id.  
prompt: The prompt/description string that describes the scenario associated with the log.  
timestamp_ns: Timestamp associated with the detections.  
track_id: Unique id assigned to each track, this is produced by your tracker.  
score: Track confidence.  
label: Integer index of the object class. This is 0 for REFERRED_OBJECTs, 1 for RELATED_OBJECTs, and 2 for OTHER_OBJECTs  
name: Object class name.  
translation_m: xyz-components of the object translation in the city reference frame, in meters.  
size: Object extent along the x,y,z axes in meters.  
yaw: Object heading rotation along the z axis.  
An example looks like this:

### Example Submission
```python
example_tracks = {
  ('02678d04-cc9f-3148-9f95-1ba66347dff9','vehicle turning left at stop sign'): [
    {
       'timestamp_ns': 315969904359876000,
       'translation_m': array([[6759.51786422, 1596.42662849,   57.90987307],
             [6757.01580393, 1601.80434654,   58.06088218],
             [6761.8232099 , 1591.6432147 ,   57.66341136],
             ...,
             [6735.5776378 , 1626.72694938,   59.12224152],
             [6790.59603472, 1558.0159741 ,   55.68706682],
             [6774.78130127, 1547.73853494,   56.55294184]]),
       'size': array([[4.315736  , 1.7214599 , 1.4757565 ],
             [4.3870926 , 1.7566483 , 1.4416479 ],
             [4.4788623 , 1.7604711 , 1.4735452 ],
             ...,
             [1.6218852 , 0.82648355, 1.6104599 ],
             [1.4323177 , 0.79862624, 1.5229694 ],
             [0.7979312 , 0.6317313 , 1.4602867 ]], dtype=float32),
      'yaw': array([-1.1205611 , ... , -1.1305285 , -1.1272993], dtype=float32),
      'name': array(['REFERRED_OBJECT', ..., 'REFERRED_OBJECT', 'RELATED_OBJECT'], dtype='<U31'),
      'label': array([ 0, 0, ... 0,  1], dtype=int32),
      'score': array([0.54183, ..., 0.47720736, 0.4853499], dtype=float32),
      'track_id': array([0, ... , 11, 12], dtype=int32),
    },
    ...
  ],
  ...
}
```

### Additional Competition Details

* Language queries are object-centric -- all correspond to some set of objects.
* Most language queries are given from the third person persective (such as "ego vehicle turning left"). The language queries given from the first-person perspective (such as "the pedestrian on the right") describe objects from the point of view of the ego vehicle.
* In the case the language query does not refer to an object (such as "raining"), the track bounding boxes should be drawn around the ego vehicle.
* Scenarios only involve objects within 50 meters from the ego vehicle and within 5 meters of a mapped road.
* Interacting objects within a scenario are at most 50 meters away from each other. 
* All referred object tracks persist for at least 3 evaluation timestamps (1.5s).
  
The ego vehicle has the following bounding box across all logs and timestamps
'translation_m': [1.422, 0, 0.25]
'size': [4.877, 2, 1.473]
'yaw': [0]

### Contact 

Any questions or discussion are welcome! Please raise an issue (preferred), or send me an email.

Cainan Davidson [crdavids@andrew.cmu.edu]

