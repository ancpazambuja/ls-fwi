
"""@package inference.py
Given a experiment id and a run id, this module loads the model and performs inference on the test data.
The main goals are:
 - analyse model's generalization capabilities;
 - analyse latent space quality;
"""


############################
# basic stuff
import os
from datetime import datetime
import time
import random
import sys
import struct
from glob import glob
import pickle

############################
# science stuff
from scipy import ndimage
from scipy import interpolate
#from skimage import io as skio
from scipy import fftpack

import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
# import cv2

#get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors0


############################
# image
from PIL import Image
import segyio


############################
# experiment tracking
import mlflow
from mlflow import MlflowClient


############################
# ml stuff
import sklearn
from sklearn.model_selection import GridSearchCV

############################
# Parallelism
#from multiprocessing import Pool
#from torch.multiprocessing.pool import ThreadPool
from multiprocessing.pool import ThreadPool

############################
# internal modules

from src.experiment_tracking.mlflow_utils import MLFlowUtils, MLFlowDataAndModelsLogger
from src.experiment_tracking.mlflow import MLFlow
from src.experiment_tracking.mlflow_skorch_callbacks import ParamLogger

from src.processing.gs_grid import InferenceGrid, GridSearchElementResult
from src.processing.task_processing import Task, BuildTasks, TaskProcessingGlobals, InferenceTaskProcessing

from src.utl.logging import Log as log

from src.utl.classes import Utils as utl
from src.utl.classes import Graph as graph

from src.visualization.data_viewer import DataAndModelsViewer as viewer

from src.preprocessing.pipeline import Preprocessing

from src.apps.command_line_interface import InferenceCLI
from src.apps.global_vars import InferenceGlobalVars

#from src.preprocessing.local_pipeline import PreprocessingPipe
#from src.processing.train_data import TrainDataInfo, TrainData, Jobs, ProductionOutlierClassification


def read_refit_run_id_attributes(uri=None, refit_experiment_name=None, refit_run_id=None):

        _, run = MLFlowUtils.create_client_and_get_run(
            uri=uri, 
            experiment_name=refit_experiment_name,
            experiment_id=MLFlowUtils.get_experiment_id_by_experiment_name(refit_experiment_name), 
            execution_id=refit_run_id,
        )

        run_data_dict = run.data.to_dictionary()

        assert run_data_dict['params'] is not None, f"Error! There is no 'params' attribute associated to the best run user parameter!"
        assert len(run_data_dict['params'].keys()) > 0, f"Error! run_data_dict['params'] has zero attributes!"
        return run_data_dict['tags'], run_data_dict['params'], run_data_dict['metrics']


def print_lib_versions(verbose=False):
    if verbose:
        print("Python version")
        print (sys.version)
        print("Version info.")
        print (sys.version_info)
        print("")
        print("Matplotlib version")
        print('{}'.format(matplotlib.__version__))
        print("")


def create_mlflow_wrapper(uri=None, experiment_name=None, verbose=False):
    mlflow_ = MLFlow(
            uri=gvars.get_experiment_tracking_uri(), 
            experiment_name=gvars.get_experiment_tracking_name(),
            verbose=True
        )

    mlflow_.info()

    return mlflow_


#########
# Create mlflow logging instance and set global vars' run name and run id.
def create_run(mlflow_=None, run_name=None):
    run_id, returned_run_name = mlflow_.init(run_name=run_name)
    gvars.set_execution_id(run_id)
    assert run_name == returned_run_name, f"Internal error! run_name ({run_name}) differs from returned_run_name ({returned_run_name})"
    if gvars.is_verbose():
        log.info(f"NEW JOB! run id: {run_id}")
        log.info(f"         run name: {run_name}")

    return run_id, returned_run_name

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == "__main__":

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # reproducibility 

    args = InferenceCLI().parse_args()
    print(args)
    gvars = InferenceGlobalVars(
        verbose_level=args.verbose_level,
        mlflow_uri=args.experiment_tracking_uri,
        mlflow_experiment=args.experiment_tracking_name,
        mlflow_experiment_id = None, # to be set after mlflow initialization
        device=args.device, 
        data=args.dataset_type,
        data_resolution=args.dataset_resolution,
        data_max_samples=args.dataset_max_samples,
        refit_run_id=args.refit_run_id,
        refit_experiment_tracking_name=args.refit_experiment_tracking_name,
        model_path=args.model_path,
        label_id=args.label_id,
        seed=args.seed,
        use_transformer=args.model_arch == "transencdec",
    )



print_lib_versions(verbose=gvars.is_verbose())
log.info(f"args = {args}")
log.info(f"gvars = {gvars}")

#########
# initialize mlflow 
mlflow_ = create_mlflow_wrapper(
            uri=gvars.get_experiment_tracking_uri(), 
            experiment_name=gvars.get_experiment_tracking_name(),
            verbose=True
        )
mlflow_.basic_init()

# read refit run parameters from mlflow
run_tags, run_params, _ = read_refit_run_id_attributes(
    uri=gvars.get_experiment_tracking_uri(),
    refit_experiment_name=gvars.get_refit_experiment_tracking_name(),
    refit_run_id=gvars.get_refit_run_id(),
)
gvars.set_attributes_from_refit_run_attributes(run_attributes=run_params) 
gvars.set_refit_run_name(run_tags.get("mlflow.runName")) 

# Create run and set global vars' run name and run id.
gvars.set_execution_name(utl.get_execution_names(global_vars=gvars)[1])
run_id, _ = create_run(mlflow_=mlflow_, run_name=gvars.get_execution_name()) 
gvars.set_execution_id(run_id)

gvars.set_experiment_tracking_id(MLFlowUtils.get_experiment_id_by_experiment_name(gvars.get_experiment_tracking_name()))

gvars.load_data()

log.line()
log.info(f"gvars = {gvars}")
log.line()

# # Preprocessing
# 
# **NOTES:**
# The preprocessing is inplace.
# 
# The last step is a out of place conversion to pytorch tensor.
preprocess_pipe = Preprocessing(verbose=True)

__, __ = preprocess_pipe.fit_transform(data=gvars.get_raw_train_data(), models=gvars.get_raw_train_models())
__, __ = preprocess_pipe.transform(data=gvars.get_raw_test_data(), models=gvars.get_raw_test_models())



#########
# Understanding the data
#   - selecting samples to view
#   - generating images in order to log them on mlflow (at the end or execution)
NSAMPLES = 3
train_samples = list()
test_samples = list()
for i in range(NSAMPLES):
    train_samples.append(random.randint(0,gvars.get_raw_train_data().get_nmb_of_elems()-1))
    test_samples.append(random.randint(0,gvars.get_raw_test_data().get_nmb_of_elems()-1))

log.info(f"TRAIN SAMPLES = {train_samples}")
log.info(f"TEST SAMPLES = {test_samples}")

train_logger = MLFlowDataAndModelsLogger(verbose=gvars.is_verbose())
train_logger.make_figs(
        indices=train_samples, 
        models=gvars.get_raw_train_models(), 
        data=gvars.get_raw_train_data().get_current_data(),
        colorbar=True, 
        res_str=gvars.get_raw_train_data().get_current_resolution_name()
    )

test_logger = MLFlowDataAndModelsLogger(verbose=gvars.is_verbose())
test_logger.make_figs(
        indices=test_samples, 
        models=gvars.get_raw_test_models(), 
        data=gvars.get_raw_test_data().get_current_data(),
        colorbar=True, 
        res_str=gvars.get_raw_train_data().get_current_resolution_name()
    )


##########
# Grid creation
#assert gvars.is_encdec_only(),f"Unexpected error! gvars.is_encdec_only must be True. gvars.is_encdec_only = {gvars.is_encdec_only()}"
grid = InferenceGrid(global_vars=gvars)
expanded_grid = grid.expand()


##########
# Define number of workers
is_gpu = torch.cuda.is_available()
if is_gpu:
    device_count = torch.cuda.device_count()
else:
    device_count = 1


###########
tasks = BuildTasks(verbose_level=1).build(expanded_grid=expanded_grid, device=gvars.get_device(), n=device_count)
log.line()
log.info(tasks)
log.line()

task_processing = TaskProcessingGlobals(
        global_vars=gvars, 
        train_data=gvars.get_raw_train_data(),
        test_data=gvars.get_raw_test_data(),
        mlflow_=mlflow_,
        verbose_level=gvars.get_verbose_level(),
        processing_class=InferenceTaskProcessing
    )

###########
# Init parallelism and start workers
with ThreadPool(min(len(tasks),device_count)) as p:
    results = p.map(task_processing, tasks)
    print(f"mse_eval  - mean, std = {results[0][0].mean()}, {results[0][0].std()}") 
    print(f"mae_eval  - mean, std = {results[0][1].mean()}, {results[0][1].std()}") 
    print(f"ssim_eval - mean, std = {results[0][2].mean()},  {results[0][2].std()}") 


# Log dataset examples
mlflow_client, run = MLFlowUtils.create_client_and_get_run(
    mlflow_=mlflow_, 
    experiment_id=gvars.get_experiment_tracking_id(),
    execution_id=gvars.get_execution_id(), 
    execution_name=gvars.get_execution_name()
)

train_title_prefix="TRAIN_" + gvars.get_data_resolution() + "res"
test_title_prefix="TEST_" + gvars.get_data_resolution() + "res" 
train_logger.log_client(mlflow_client=mlflow_client, run_id=run.info.run_id, title_prefix=train_title_prefix)
test_logger.log_client(mlflow_client=mlflow_client, run_id=run.info.run_id, title_prefix=test_title_prefix)


log.info("FINALIZANDO")





