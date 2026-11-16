

"""@package task_processing.py
Code for thread processing.
"""


############################
# basic stuff
import os
from datetime import datetime
import time
import random
import sys
from glob import glob
import pickle
import copy

import torch.nn.functional as F
    
import numpy as np
import seaborn as sns

############################
# experiment tracking
import mlflow
from mlflow import MlflowClient


############################
# ml stuff
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
import torch
import torch.nn as nn
from torch.optim import SGD, Adam #, Adagrad, Adamax, NAdam, Adadelta, RMSprop
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from skorch import dataset
from skorch.callbacks import MlflowLogger, Checkpoint


############################
# internal modules

from src.experiment_tracking.mlflow_utils import MLFlowUtils, MLFlowDataAndModelsLogger
from src.experiment_tracking.mlflow import MLFlow
from src.experiment_tracking.mlflow_skorch_callbacks import ParamLogger

from src.processing.gs_grid import GridSearchElementResult
from src.processing.ml_models_builder import SkorchEncoderDecoderModelBuilder, ScoredSkorchModel, SkorchGeneratorInversionBuilder, SkorchGeneratorModelBuilder, SkorchEncoderModelBuilder
from src.processing.ml_models import EncoderDecoder 
from src.processing.ml_models_utils import MLModelsUtils

from src.utl.logging import Log as log

from src.utl.classes import Utils as utl
from src.utl.classes import Graph as graph

from src.visualization.data_viewer import DataAndModelsViewer as viewer

from src.preprocessing.pipeline import Preprocessing

from src.apps.command_line_interface import TrainCLI
from src.apps.global_vars import GlobalVars

def get_result(index=None, gs=None):
    print(f"index = {index}")
    print(gs)
    return GridSearchElementResult(
        index=index,
        mean_train_score = -gs.cv_results_['mean_train_score'][0],
        std_train_score = -gs.cv_results_['std_train_score'][0],
        mean_valid_score = -gs.cv_results_['mean_test_score'][0],
        std_valid_score = -gs.cv_results_['std_test_score'][0],
    )

class Task:
    """
    The task to be processed.
    """
    def __init__(self, id=0, device='cpu', idxs=None, items=None, verbose_level=0):
        self.verbose_level = verbose_level
        self.verbose = True if verbose_level > 0 else False
        self.device = device
        self.id = id # task index inside global grid.

        assert len(idxs) == len(items), f"'idxs' and 'items' must have the same size! #idxs={len(idxs)} and #items={len(items)}"
        self.items = items
        self.global_idxs = idxs

    def get_id(self):
        return self.id

    def get_device(self):
        return self.device

    def set_device(self, device='cpu'):
        self.device = device

    def get_item_global_idx(self, idx):
        assert idx >= 0 and idx < self.get_n_items(), f'Invalid index! (idx = {idx})'
        return self.global_idxs[idx]

    def get_item(self, idx):
        assert idx >= 0 and idx < self.get_n_items(), f'Invalid index! (idx = {idx})'
        return self.items[idx]

    def get_all_items_global_idxs(self):
        return self.global_idxs

    def get_all_items(self):
        return self.items

    def get_n_items(self):
        return len(self.items)

class BuildTasks:
    """
    The task to be processed.
    """
    def __init__(self, verbose_level=None):
        self.verbose_level = verbose_level
        self.verbose = True if verbose_level > 0 else False

    def build(self, expanded_grid=None, device='cpu', n=1):
        '''
        n is the number of workers.
        '''
        tasks = list()
        l = len(expanded_grid)
        log.info(f"l = {l}")
        for i in range(0,min(l,n)): 
            idxs = range(i,l,n) #using rounding robin in order to try to keep work balance
            log.info(f"idxs = {idxs}")
            log.info(f"idxs.len = {len(idxs)}")
            items= list()
            for j in range(0,len(idxs)):
                #print(expanded_grid[j])
                items.append(expanded_grid[idxs[j]])
            #for j in idxs:
            #    items.append(expanded_grid[j])
            tasks.append(Task(id=i, device=device, idxs=idxs, items=items, verbose_level=self.verbose_level))

        # treat for 'gpu:all' case
        if  device == 'cuda:all':
            for i in range(0,len(tasks)):
                tasks[i].set_device(f'cuda:{i}')

        if self.verbose:
            log.info(f"Number of tasks = {len(tasks)}")
            log.info(f"----- TASKS -----")
            for i, t in enumerate(tasks):
                log.info(f"task number = {i}")
                log.info(f"id          = {t.id}")
                log.info(f"global idxs = {t.global_idxs}")
                log.info(f"device      = {t.device}")
                log.line()

        return tasks

class GridSearchProcessing:
    """
    Execute the grid search with cross validation using scilearn framework.
    """
    def __init__(self, estimator=None, grid=None, cv=5, verbose_level=0):
        self.verbose_level = verbose_level
        self.verbose = True if verbose_level > 0 else False

        self.estimator = estimator
        self.grid = grid
        self.cv = cv

        self.x_train = None
        self.y_train = None

        self.gs = GridSearchCV(
                estimator=self.estimator, 
                param_grid=self.grid, 
                cv=self.cv, 
                error_score='raise', 
                return_train_score=True, 
                n_jobs=1, 
                refit=False, 
                verbose=True
            )

    def fit(self, x_train=None, y_train=None):
        self.x_train = x_train
        self.y_train = y_train

        if self.verbose:
            log.info(f"[ START ] Grid search with cv = {self.cv}.")

        results = self.gs.fit(x_train, y_train)

        if self.verbose:
            log.info(f"[ END ] Grid search")

        return self.gs, results




class TaskProcessing:
    def __init__(self, task_processing_globals=None, task=None):
        self.g = task_processing_globals
        self.task = task

        self.model_data_size = None
        self.model = None
        self.cropped_x = None
        self.cropped_y = None
        self.cropped_x_test = None
        self.cropped_y_test = None
        self.device = None
        self.device = torch.device(self.task.get_device()) # construct an object of class torch.device

    def run(self):
        assert str(self.device) == self.task.get_device(), f"Logical internal error"
        utl.set_seed(self.g.gvars.get_seed())

        # find model architecture compatible sizes
        self.model_data_size = self.get_model_data_size()
        if self.g.verbose:
            log.info(f"model_data_size: {self.model_data_size}")

        # set the target device
        if torch.cuda.is_available():
            print (f"-----------------------")
            print (f"cuda current device : {torch.cuda.current_device()}")
            print (f"self.device : {self.device}")
            print (f"-----------------------")
            torch.cuda.device(self.device)
            print (f"[ thread {self.task.get_id()}cuda current device : {torch.cuda.current_device()}")
            torch.cuda.empty_cache()
            self.stream = torch.cuda.Stream(device=self.device)
        #self.set_torch_device(self.task.get_device())
        
        with torch.cuda.stream(self.stream):

            # deal with mlflow client and run
            mlflow_client, run = MLFlowUtils.create_client_and_get_run(
                mlflow_=self.g.mlflow, 
                experiment_id=self.g.gvars.get_experiment_tracking_id(),
                execution_id=self.g.gvars.get_execution_id(), 
                execution_name=self.g.gvars.get_execution_name()
            )

            # create callbacks for trainning
            gs_cbs, _, lr_scheduler, patience = self.create_callbacks(
                mlflow_client=mlflow_client, 
                run=run, 
                is_one_step_training=self.g.gvars.is_one_step_training()
            )

            # create model
            print (f"-----------------------")
            print (f"[ thread {self.task.get_id()} cuda current device : {torch.cuda.current_device()}")
            self.model = self.create_model(cb=gs_cbs, device=self.device)

            # create a view of the data from self.model_data_size
            self.cropped_x, self.cropped_y, _, _ = self.crop_data(device=None) #self.device) # Testar com True

            if self.g.verbose:
                log.info(f"[ START ] Grid search for task id {self.task.get_id()} with {self.task.get_n_items()} hyper-parameters instances.")

            # Deal with mlflow run setup
            #with mlflow.start_run(run_id=self.g.gvars.get_execution_id(), log_system_metrics=True) as run:
            best_result = GridSearchElementResult()
            for i,t in enumerate(self.task.get_all_items()):
                log.info(f"[ {i} ]")
                gscv = GridSearchProcessing(
                    estimator=self.model, 
                    grid=[t], #task.get_item(i), 
                    verbose_level=self.g.verbose_level,
                    cv=self.g.cv, 
                    )
                gs, results = gscv.fit(x_train=self.cropped_x, y_train=self.cropped_y)
                new_result =  get_result(index=self.task.get_item_global_idx(i),gs=gs)
                updated = best_result.update(r=new_result)
                if self.g.verbose:
                    log.info(f"New best value! global index:{best_result.index}, mean_valid_score={best_result.mean_valid_score} ")
                #print(f"gs.param_grid = {gs.param_grid}")
                
                torch_model = self.model.module_
                model_size=sum(p.numel() for p in torch_model.parameters())
                # log in mlflow the result
                MLFlowUtils.log_results(
                    mlflow_client=mlflow_client, experiment_id=self.g.mlflow.get_experiment_id(), run=run, base_id=self.task.get_item_global_idx(i),
                    gridsearch=gs, model_size=model_size, lr_scheduler=lr_scheduler,
                    patience=patience, seed=self.g.gvars.get_seed(),
                    test_perc=self.g.gvars.get_test_perc(), train_indexes=self.g.gvars.get_train_indexes(), test_indexes=self.g.gvars.get_test_indexes()

                ) #, tags={}, log_only_best=False)
                # save checkpoint
                # save()
                #update best result
                #best_result = ....
        if self.g.verbose:
            log.info(f"[ END ] Grid search finished.")

        return best_result

    def crop_data(self, device=None):

        if self.g.verbose_level > 0: 
            log.info(f"DEVICE = {device}")
    
        # X - expected shape (batch, nchannels, nz, nx)
        nz_cut = self.model_data_size['nz_cut']
        nx_cut = self.model_data_size['nx_cut']
        x_train = self.g.train_data.get_current_data().get_X()[:,:,:nz_cut,:nx_cut]
        x_test = self.g.test_data.get_current_data().get_X()[:,:,:nz_cut,:nx_cut]
        if self.g.verbose_level > 0: 
            log.info(f"x_train.shape = {x_train.shape}")
            log.info(f"x_test.shape = {x_test.shape}")
        
        # y_train - expected shape (batch, nz, nx)
        y_train = self.g.train_data.get_current_data().get_Y()[:,:nz_cut,:nx_cut]
        y_test = self.g.test_data.get_current_data().get_Y()[:,:nz_cut,:nx_cut]
        if self.g.verbose_level > 0: 
            log.info(f"y_train.shape = {y_train.shape}    --- original shape")
            log.info(f"y_test.shape = {y_test.shape}    --- original shape")

        # y_train - reshaped: from (batch, nz, nx) to (batch, 1, nz, nx)
        y_shape = y_train.shape
        y_new_shape = (y_shape[0], 1, y_shape[1], y_shape[2])
        y_train = y_train.reshape(y_new_shape)

        y_test_shape = y_test.shape
        y_test_new_shape = (y_test_shape[0], 1, y_test_shape[1], y_test_shape[2])
        y_test = y_test.reshape(y_test_new_shape)

        assert y_train.shape[1:-1] == y_test.shape[1:-1], f"Internal Error! y_train and y_test must have the same shape (y_train.shape = {y_train.shape}, y_test.shape = {y_test.shape})"


        if self.g.verbose_level > 0: 
            log.info(f"y_train.shape = {y_train.shape}    --- after reshaping")
            log.info(f"y_test.shape = {y_test.shape}    --- after reshaping")
        
        if self.g.verbose_level > 0: 
            log.info(f"x_train,y_train current devices = {x_train.device}, {y_train.device}")
            log.info(f"x_test,y_test current devices = {x_test.device}, {y_test.device}")

        if device is not None:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            if self.g.verbose_level > 0: 
                log.info(f"x_train,y_train after to(device) = {x_train.device}, {y_train.device}")
                log.info(f"x_test,y_test after to(device) = {x_test.device}, {y_test.device}")

        return x_train, y_train, x_test, y_test

    def get_model_data_size(self):
        return utl.get_model_data_size(self.g.train_data, depth=5)

    #def set_torch_device(self, device):
    #    self.device = torch.device(device)
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()

    def create_callbacks(self, mlflow_client=None, run=None, is_one_step_training=True, is_inversion=False):
        return utl.get_train_callbacks(global_vars=self.g.gvars, mlflow_client=mlflow_client, run=run,
            is_one_step_training=is_one_step_training, is_inversion=is_inversion)

    def create_model(self, cb=None, device=None):

        if self.g.verbose_level > 0:
            n_channels = self.model_data_size['nchannels']
            log.info(f"N_CHANNELS = {n_channels}")
            log.info(f"is_transformer = {self.g.gvars.is_transformer()}")
            log.info(f"task device = {self.task.get_device()}")
            log.info(f"device = {device}")
            log.info(f"device type = {type(device)}")
            log.info(f"cuda current device = {torch.cuda.current_device()}")
            log.info(f"is_one_step_training = {self.g.gvars.is_one_step_training()}")

        # size of the attention size
        if self.g.gvars.get_data_type() == "shallow":
            attention_img_size=(11,28) #10x28=280, sequence length quando patches.size=(1,1)
        elif self.g.gvars.get_data_type() == "deep": 
            attention_img_size=(21, 25) #21x25 = 525, sequence length quando patches.size=(1,1)
        else:
            raise ValueError

        model = SkorchEncoderDecoderModelBuilder(verbose=self.g.verbose).build(
            seed=self.g.gvars.get_seed(),
            max_epochs=self.g.gvars.get_max_epochs(),
            conv_kernel_size=self.g.gvars.get_conv_filter_size(),
            conv_dilation_size=self.g.gvars.get_conv_dilation_size(),
            transposed_conv_kernel_size=self.g.gvars.get_transposed_conv_filter_size(),
            use_transformers=self.g.gvars.is_transformer(),
            unembed_type=self.g.gvars.get_unembed_type(),

            callbacks=cb,

            device=device,
            batch_size=16,
            #criterion=nn.SSIM(),
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam,
            lr=0.0001,
            in_channels=n_channels, out_channels=1, n_filters_init=16,
            pool_kernel_size=(2,2), pool_stride=(2,2),
            batch_normalization=True, dropout=0.2,
            #img_size=(672,800),
            attention_img_size=attention_img_size, #21x25=525 == sequence length quando patches.size=(1,1)
            attention_in_channels=512,  # n_channels of the last encoding feature map
            attention_hidden_size=512, # D, embedding size 
            attention_num_heads=16,
            attention_num_layers=12,
            attention_mlp_dim=4096,
            vis=False,
            )

        if self.g.verbose:
            model.initialize()
            input_size = ( 
                1, 
                self.model_data_size['nchannels'],
                self.model_data_size['nz_cut'], 
                self.model_data_size['nx_cut'],
            )
            print(model.module_.summary(input_size=input_size))

        return model



class RefitTaskProcessing(TaskProcessing):
    def __init__(self, task_processing_globals=None, task=None):
        super().__init__(task_processing_globals=task_processing_globals, task=task)

    def run(self):
        assert str(self.device) == self.task.get_device(), f"Logical internal error"
        utl.set_seed(self.g.gvars.get_seed())

        # find model architecture compatible sizes
        self.model_data_size = self.get_model_data_size()
        if self.g.verbose:
            log.info(f"model_data_size: {self.model_data_size}")
            log.info(f"[ START ] Refit using taks id {self.task.get_id()} with {self.task.get_n_items()} hyper-parameters instances.")

        # set the target device
        if torch.cuda.is_available():
            print (f"-----------------------")
            print (f"cuda current device : {torch.cuda.current_device()}")
            print (f"self.device : {self.device}")
            print (f"-----------------------")
            torch.cuda.device(self.device)
            print (f"[ thread {self.task.get_id()}cuda current device : {torch.cuda.current_device()}")
            torch.cuda.empty_cache()
            self.stream = torch.cuda.Stream(device=self.device)
        #self.set_torch_device(self.task.get_device())
        
        with torch.cuda.stream(self.stream):

            # deal with mlflow client and run
            mlflow_client, run = MLFlowUtils.create_client_and_get_run(
                mlflow_=self.g.mlflow, 
                experiment_id=self.g.gvars.get_experiment_tracking_id(),
                execution_id=self.g.gvars.get_execution_id(), 
                execution_name=self.g.gvars.get_execution_name()
            )

            # create callbacks for trainning
            _, refit_cbs, lr_scheduler, patience = self.create_callbacks(
                mlflow_client=mlflow_client, 
                run=run,
                is_one_step_training=self.g.gvars.is_one_step_training(),
            )
            log.info(f"refit_cbs = {refit_cbs}")

            # create model
            print (f"-----------------------")
            print (f"[ thread {self.task.get_id()} cuda current device : {torch.cuda.current_device()}")
            self.model = self.create_model(cb=refit_cbs, device=self.device, grid=self.task.get_item(0))

            # create a view of the data from self.model_data_size
            self.cropped_x, self.cropped_y, self.cropped_x_test, self.cropped_y_test = self.crop_data(device=None)

            # Refit 
            # Note: model must be parameterized to:
            #   i) splits data into train and validation, 
            #   ii) performs checkpoint and early stopping,
            #   iii) load the best model at the end ('load_best'=True) 
            self.model.fit(self.cropped_x, self.cropped_y) 

            # Evaluate
            # predict (it worked because checkpoint was set with 'load_best=true')
            self.model.module_.eval() # it might not be necessary, because skorch probably does it internally in predict
            y_hat = self.model.predict(self.cropped_x_test)
            # use mse (loss) and ssim (index) metrics
            mae, _  = self.mae(self.cropped_y_test, y_hat)
            mse, _  = self.mse(self.cropped_y_test, y_hat)
            ssim, ssim_qmap = self.ssim(self.cropped_y_test, y_hat)
            if self.g.verbose:
                log.info(f"MSE mean over test set is: {mse.mean()}")
                log.info(f"SSIM mean over test set is: {ssim.mean()}")
            self.log_evaluation_results(
                mlflow_client=mlflow_client, run=run, 
                mae=mae, mse=mse, ssim=ssim, ssim_qmap=ssim_qmap, 
                y=self.cropped_y_test, y_hat=y_hat
            )

            torch_model = self.model.module_
            model_size=sum(p.numel() for p in torch_model.parameters())
            self.log_parameters(mlflow_client=mlflow_client, run=run, 
                params=self.task.get_item(0), 
                model_size=model_size, lr_scheduler=lr_scheduler,
                patience=patience, seed=self.g.gvars.get_seed(),
                test_perc=self.g.gvars.get_test_perc(), train_indexes=self.g.gvars.get_train_indexes(), test_indexes=self.g.gvars.get_test_indexes()
            )

        if self.g.verbose:
            log.info(f"[ END ] Refit finished.")

        return mse, ssim

    #def set_torch_device(self, device):
    #    self.device = torch.device(device)
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()

    def create_model(self, cb=None, device=None, grid=None):

        if self.g.verbose_level > 0:
            n_channels = self.model_data_size['nchannels']
            log.info(f"N_CHANNELS = {n_channels}")
            log.info(f"is_transformer = {self.g.gvars.is_transformer()}")
            log.info(f"task device = {self.task.get_device()}")
            log.info(f"device = {device}")
            log.info(f"device type = {type(device)}")
            log.info(f"cuda current device = {torch.cuda.current_device()}")
            log.info(f"type of grid = {type(grid)}")
            log.info(f"one_step_training = {self.g.gvars.is_one_step_training()}")
            log.info(f"grid = {grid}")


        model = ScoredSkorchModel(
            module=EncoderDecoder,
            device=device,
            callbacks=cb,
            seed=self.g.gvars.get_seed(),
            **grid
        )

        if self.g.verbose:
            model.initialize()
            input_size = ( 
                1, 
                self.model_data_size['nchannels'],
                self.model_data_size['nz_cut'], 
                self.model_data_size['nx_cut'],
            )
            print(model.module_.summary(input_size=input_size))

        return model

    def mae(self, y, y_hat):
        return self.evaluate(y, y_hat, metric=nn.L1Loss(), qmap=False)

    def mse(self, y, y_hat):
        return self.evaluate(y, y_hat, metric=nn.MSELoss(), qmap=False)

    def ssim(self, y, y_hat):
        return self.evaluate(y, y_hat, metric=utl.ssim, qmap=True)

    def evaluate(self, y, y_hat, metric=None, qmap=True):
        assert metric is not None, f"Internal error! A metric must be passed."
        assert y.shape == y_hat.shape, f"Internal error! y and y_hat must have the same size. (y.shape = {y.shape}, y_hat.shape = {y.shape})"

        metric_individual = np.zeros((y.shape[0]))
        if qmap:
            quality_map = np.zeros((y.shape[0],y.shape[2],y.shape[3]))
        else:
            quality_map = None


        for k in range(0,y.shape[0]):
            if qmap:
                metric_individual[k], quality_map[k] = metric(y[k,0], torch.from_numpy(y_hat[k,0]))
            else:
                metric_individual[k] = metric(y[k,0], torch.from_numpy(y_hat[k,0]))

        return  metric_individual, quality_map

    def log_evaluation_results(self, mlflow_client=None, run=None, label="", mae=None, mse=None, ssim=None, ssim_qmap=None, y=None, y_hat=None):
        assert y.shape == y_hat.shape, f"y and y_hat must be the same size! y.shape = {y.shape} and y_hat.shape = {y_hat.shape}"
        assert y.shape[0] == len(mae), f"y and mse must be the same size! len(y) = {y.shape[0]} and len(mse) = {len(mae)}"
        assert y.shape[0] == len(mse), f"y and mse must be the same size! len(y) = {y.shape[0]} and len(mse) = {len(mse)}"
        assert y.shape[0] == len(ssim), f"y and ssim must be the same size! len(y) = {y.shape[0]} and len(ssim) = {len(ssim)}"

        run_id = run.info.run_id

        for t,v in enumerate(mae): 
            mlflow_client.log_metric(run_id=run_id, key=label+"_mae_values", value=v, step=t)
        mlflow_client.log_metric(run_id, label+"_mae_mean_test_set", mae.mean())
        mlflow_client.log_metric(run_id, label+"_mae_std_test_set", mae.std())
        mlflow_client.log_metric(run_id, label+"_mae_min_test_set", mae.min())
        mlflow_client.log_metric(run_id, label+"_mae_max_test_set", mae.max())

        for t,v in enumerate(mse): 
            mlflow_client.log_metric(run_id=run_id, key=label+"_mse_values", value=v, step=t)
        mlflow_client.log_metric(run_id, label+"_mse_mean_test_set", mse.mean())
        mlflow_client.log_metric(run_id, label+"_mse_std_test_set", mse.std())
        mlflow_client.log_metric(run_id, label+"_mse_min_test_set", mse.min())
        mlflow_client.log_metric(run_id, label+"_mse_max_test_set", mse.max())

        for t,v in enumerate(ssim):
            mlflow_client.log_metric(run_id=run_id, key=label+"_ssim_values", value=v, step=t)
        mlflow_client.log_metric(run_id, label+"_ssim_mean_test_set", ssim.mean())
        mlflow_client.log_metric(run_id, label+"_ssim_std_test_set", ssim.std())
        mlflow_client.log_metric(run_id, label+"_ssim_min_test_set", ssim.min())
        mlflow_client.log_metric(run_id, label+"_ssim_max_test_set", ssim.max())
        
        y2 = y.reshape((y.shape[0], y.shape[2], y.shape[3]))
        y_hat2 = y_hat.reshape((y_hat.shape[0], y_hat.shape[2], y_hat.shape[3]))

        yy_mae_figs = viewer.yy_view(y=y2, y_hat=y_hat2, loss=mae, samples=list(range(len(mae))),
                 vmin=0, vmax=1, colorbar=True, title=label+" MAE - Evaluation")

        yy_mse_figs = viewer.yy_view(y=y2, y_hat=y_hat2, loss=mse, samples=list(range(len(mse))),
                 vmin=0, vmax=1, colorbar=True, title=label+" MSE - Evaluation")

        yyq_ssim_figs = viewer.yyq_view(y=y2, y_hat=y_hat2, qmap=ssim_qmap, loss=ssim, samples=list(range(len(ssim))),
                 vmin=0, vmax=1, colorbar=True, title=label+" SSIM - Evaluation")

        mlflow_client.log_figure(run_id, yy_mae_figs, label+"_MAE_EVAL.png")
        mlflow_client.log_figure(run_id, yy_mse_figs, label+"_MSE_EVAL.png")
        mlflow_client.log_figure(run_id, yyq_ssim_figs, label+"_SSIM_EVAL.png")

    def log_parameters(self, mlflow_client=None, run=None, params=None, model_size=None,
                       lr_scheduler=None, patience=None, seed=None, 
                       test_perc=None, train_indexes=None, test_indexes=None):
        run_id=run.info.run_id
        mlflow_client.log_param(run_id=run_id, key="model_size", value=model_size)
        mlflow_client.log_param(run_id=run_id, key="lr_scheduler", value=lr_scheduler)
        mlflow_client.log_param(run_id=run_id, key="patience", value=patience)
        mlflow_client.log_param(run_id=run_id, key="_x_seed", value=seed)
        mlflow_client.log_param(run_id=run_id, key="_x_test_perc", value=test_perc)
        mlflow_client.log_param(run_id=run_id, key="_x_train_indexes", value=train_indexes)
        mlflow_client.log_param(run_id=run_id, key="_x_test_indexes", value=test_indexes)
        MLFlowUtils.log_parameters_by_run(mlflow_client=mlflow_client, run=run, params=params)


class TaskProcessingGlobals:
    """
    Process the task in thread-safe manner. 
    'gvars', 'train_data' and 'expanded_grid' will not be changed internally
    and must not be changed from outside after threads have started
    """
    def __init__(self, global_vars=None, train_data=None, test_data=None, cv=None,  mlflow_=None, verbose_level=None, processing_class=TaskProcessing):

        # global attributes - keep them constant for thread-safety, don't change them
        self.gvars = global_vars
        self.train_data = train_data
        self.test_data= test_data

        # global attributes - already initialized, 
        # but can be used to log results
        self.mlflow = mlflow_

        # private attribute - free to use, can be changed
        self.verbose_level = verbose_level
        self.verbose = True if verbose_level > 0 else False
        self.cv = cv
        self.processing_class = processing_class

        self.model_data_size = None
        self.task = None
        self.model = None

    def __call__(self, task):
        processing_class = self.processing_class 
        task_processing = processing_class(task_processing_globals=self, task=task)
        return task_processing.run()
    




class InferenceTaskProcessing(RefitTaskProcessing):
    def __init__(self, task_processing_globals=None, task=None):
        super().__init__(task_processing_globals=task_processing_globals, task=task)
        self.generator = None
        self.encoder = None

    def run(self):
        assert str(self.device) == self.task.get_device(), f"Logical internal error"
        utl.set_seed(self.g.gvars.get_seed())

        # find model architecture compatible sizes
        self.model_data_size = self.get_model_data_size()
        if self.g.verbose:
            log.info(f"model_data_size: {self.model_data_size}")
            log.info(f"[ START ] Refit using taks id {self.task.get_id()} with {self.task.get_n_items()} hyper-parameters instances.")

        # set the target device
        if torch.cuda.is_available():
            print (f"-----------------------")
            print (f"cuda current device : {torch.cuda.current_device()}")
            print (f"self.device : {self.device}")
            print (f"-----------------------")
            torch.cuda.device(self.device)
            print (f"[ thread {self.task.get_id()} cuda current device : {torch.cuda.current_device()}")
            torch.cuda.empty_cache()
            self.stream = torch.cuda.Stream(device=self.device)
        #self.set_torch_device(self.task.get_device())
        
        with torch.cuda.stream(self.stream):

            # deal with mlflow client and run
            mlflow_client, run = MLFlowUtils.create_client_and_get_run(
                mlflow_=self.g.mlflow, 
                experiment_id=self.g.gvars.get_experiment_tracking_id(),
                execution_id=self.g.gvars.get_execution_id(), 
                execution_name=self.g.gvars.get_execution_name()
            )

            # Create models
            print (f"-----------------------")
            print (f"[ thread {self.task.get_id()} cuda current device : {torch.cuda.current_device()}")
            log.debug("Calling create_model with device and task")
            log.line()
            log.info(f"grid = {self.task.get_item(0)['skorch_model_params']}")
            log.line()
            self.model, self.encoder, self.generator = self.create_model(device=self.device, grid=self.task.get_item(0))

            # Prepare datasets
            self.cropped_x, self.cropped_y, self.cropped_x_test, self.cropped_y_test = self.crop_data(device=None)

            # Evaluate and log results in mlflow
            mse, mae, ssim, _ = self.eval_autoencoder(mlflow_client, run, self.model)
            mse, mae, ssim, _ = self.eval_generator(mlflow_client, run, self.generator, self.encoder)

            p_diff = self.eval_latent_space_smoothness(mlflow_client, run, self.generator, self.encoder,num_interpolations=15)
            _ = self.eval_latent_space_smoothness(mlflow_client, run, self.generator, self.encoder,num_interpolations=15, extrapolate=True)

            # Follow along the latent direction of the maximum difference between two closest samples in latent space
            self.eval_along_latent_direction( mlflow_client, run, self.generator, self.encoder, self.predict(self.generator, p_diff["max"].unsqueeze(0)), 
                    num_interpolations=6, alpha_min=0.0, alpha_max=2.0, name="global_palette_diffmax_line", global_palette=True)

            # Follow along the latent direction of closest, farthest and random models from test set
            self.eval_along_latent_direction( mlflow_client, run, self.generator, self.encoder, self.cropped_x_test[0:1], 
                    num_interpolations=6, alpha_min=0.0, alpha_max=2.0, name="global_palette_line", global_palette=True)
            self.eval_along_latent_direction( mlflow_client, run, self.generator, self.encoder, self.cropped_x_test[0:1], 
                    num_interpolations=6, alpha_min=0.0, alpha_max=2.0, name="line", global_palette=False)

            # Follow a random latent vector direction (not from the test set) and evaluate the generator along this direction
            random_latent_vector = torch.rand_like(self.cropped_x_test[0:1])
            self.eval_along_latent_direction( mlflow_client, run, self.generator, self.encoder, random_latent_vector,
                    num_interpolations=6, alpha_min=0.0, alpha_max=2.0, name="global_palette_line_random_latent", global_palette=True)
            self.eval_along_latent_direction( mlflow_client, run, self.generator, self.encoder, random_latent_vector,
                    num_interpolations=6, alpha_min=0.0, alpha_max=2.0, name="line_random_latent", global_palette=False)

            # Placeholder for actual evaluation results   
            # mse, mae, ssim = np.random.rand(3), np.random.rand(3), np.random.rand(3)

            # Log parameters in mlflow
            self.log_params(mlflow_client, run, self.model, lr_scheduler=None, patience=None)


        if self.g.verbose:
            log.info(f"[ END ] Prediction finished.")

        return mse, mae, ssim




    def _build_latent_vectors_and_cosine_distance( self, encoder=None, X=None):
        """
        Find the two closest samples in latent space using cosine distance.
        """
    
        latent_vectors = encoder.predict( X )
    
        latent_vectors = torch.as_tensor( latent_vectors, dtype=torch.float32 )
    
        latent_vectors_flat = latent_vectors.reshape( latent_vectors.shape[0], -1 )
    
        latent_vectors_flat = F.normalize( latent_vectors_flat, p=2, dim=1 )
    
        cosine_distance = ( 1.0 - latent_vectors_flat @ latent_vectors_flat.T )

        return latent_vectors, cosine_distance


    def _find_pairs(self, distance_matrix=None, strategy="min"):


        if strategy == "min": 
            pair_idx = distance_matrix.clone().fill_diagonal_( float("inf") ).argmin().item()
        elif strategy == "max": 
            pair_idx = distance_matrix.argmax().item()
        elif strategy == "random":
            pair_idx = np.random.choice(distance_matrix.shape[0] * distance_matrix.shape[1])
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of ['min', 'max', 'random'].")
    
        i, j = np.unravel_index( pair_idx, distance_matrix.shape )

        return i,j

    def _generate_latent_line( self, generator, p, num_interpolations=10, alpha_min=0.0, alpha_max=2.0,):
        """
        Generate latent vectors along the line
        supported by latent vector p:
    
            z(alpha) = alpha * p
        """

        assert alpha_min < 1., f"alpha_min must be less than 1.0, but got {alpha_min}"
        assert alpha_max > 1., f"alpha_max must be greater than 1.0, but got {alpha_max}"

        alphas = torch.cat([
            torch.linspace(-alpha_max, alpha_min, num_interpolations),
            torch.linspace(alpha_min, 1.0, num_interpolations)[1:],
            torch.linspace(1.0, alpha_max, num_interpolations)[1:]
            ]) 
    
        line_latents = torch.stack( [ alpha * p for alpha in alphas ], dim=0)
    
        generated_outputs = generator.predict( line_latents.cpu().numpy())
    
        return ( line_latents, generated_outputs, alphas,)    

    def _interpolate_latent_points( self, generator, latent1, latent2, num_interpolations=10, extrapolate=False ):
        """
        Interpolate between two latent vectors and generate
        the corresponding velocity models.
        """
        if extrapolate: 
            min_v = -1.0
            max_v = 2.0
        else:
            min_v = 0.0
            max_v = 1.0  

        alphas = torch.linspace( min_v, max_v, num_interpolations)
        interpolated_latents = torch.stack( [ latent1 + (latent2-latent1)*alpha for alpha in alphas ], dim=0)

        #alphas = torch.linspace( 0.0, 1.0, num_interpolations)
        #interpolated_latents = torch.stack( [ (1.0 - alpha) * latent1 + alpha * latent2 for alpha in alphas ], dim=0)
    
        generated_outputs = generator.predict( interpolated_latents.cpu().numpy())

        return ( interpolated_latents, generated_outputs, alphas,)


    def _plot_zy( self, interpolated_latents, full_interpolated_latents, generated_outputs, label, global_palette=False):
        """
        Create a single figure showing
        z.mean, z.std, z.full, y_hat and y.
        """

        z_flat = interpolated_latents.reshape(interpolated_latents.shape[0], -1)
        vrange = torch.quantile( z_flat.abs(), q=0.99, dim=1)
        z_min = -vrange
        z_max = vrange

        # torch tensor
        #z_min = interpolated_latents.amin(dim=(1, 2, 3))
        #z_max = interpolated_latents.amax(dim=(1, 2, 3)) / 4.

        log.info(f"z_min = {z_min}")
        log.info(f"z_max = {z_max}")

        y_min = generated_outputs.amin(dim=(1, 2, 3))
        y_max = generated_outputs.amax(dim=(1, 2, 3))

        log.info(f"y_min = {y_min}")
        log.info(f"y_max = {y_max}")

        fig = viewer.zy_view( z=interpolated_latents.detach().cpu().numpy(), 
            z_full=full_interpolated_latents.detach().cpu().numpy(), 
            y=generated_outputs.detach().cpu().numpy(),
            z_min=z_min, z_max=z_max, y_min=y_min, y_max=y_max,global_palette=global_palette,
            samples=list(
                range(interpolated_latents.shape[0])
            ),
            colorbar=True,
            title=label,
        )
    
        return fig



    def _latent_to_full_image(self, latents):
        """
        Convert
    
            (N, 512, H, W)
    
        into
    
            (N, H*32, W*16)
    
        where each spatial position (i,j)
        is expanded into a 32x16 tile
        containing the 512 latent values.
    
        Parameters
        ----------
        latents : torch.Tensor
            Shape (N, 512, H, W)
    
        Returns
        -------
        torch.Tensor
            Shape (N, H*32, W*16)
        """
    
        assert torch.is_tensor(latents), f"Expected torch.Tensor, got {type(latents)}"
    
        N, C, H, W = latents.shape
    
        assert C == 512, f"Expected 512 channels but got {C}"
    
        # (N,512,H,W) # -> # (N,32,16,H,W)
        z = latents.reshape( N, 32, 16, H, W)
    
        # (N,32,16,H,W) # -> # (N,H,32,W,16) 
        z = z.permute( 0, 3, 1, 4, 2)
    
        # (N,H,32,W,16) # -> # (N,H*32,W*16)
        z = z.reshape( N, H * 32, W * 16)
    
        return z


    def eval_along_latent_direction( self, mlflow_client, run, generator, encoder, x, global_palette=False,
                                     num_interpolations=10, alpha_min=0.0, alpha_max=2.0, name="latent_direction",):
        """
        Evaluate decoder behaviour along the line
        generated by latent vector p.
    
            z(alpha) = alpha * p
        """
        p = self.predict(encoder, x)
        if p.ndim == 4 and p.shape[0] == 1:
            p = p.squeeze(0)
        p = torch.as_tensor( p, dtype=torch.float32 )

        log.info(f"Latent vector p shape: {p.shape}, min: {p.min()}, max: {p.max()}")
    
        ( line_latents, generated_outputs, alphas,) = self._generate_latent_line(
            generator=generator,
            p=p,
            num_interpolations=num_interpolations,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
    
        label = f"Latent interpolation (alpha in [{alpha_min:.2f}, {alpha_max:.2f}])"
        z_full = self._latent_to_full_image( line_latents)
        fig = self._plot_zy( line_latents, z_full, generated_outputs, label, global_palette=global_palette)
        mlflow_client.log_figure( run.info.run_id, fig, f"{name}.png")
    
        log.info(
            f"Generated {num_interpolations} latent points "
            f"along vector p for "
            f"alpha in [{alpha_min}, {alpha_max}]"
        )


    def _eval_by_strategy(self, mlflow_client, run, generator, num_interpolations, strategy, latent_vectors, distance_matrix, extrapolate):

        if extrapolate:
            extrapolate_label = "_extrapolate"
        else:
            extrapolate_label = "_interpolate"

        idx1, idx2 = self._find_pairs( distance_matrix=distance_matrix, strategy=strategy)
        log.info(
            f"Closest latent samples: "
            f"idx1={idx1}, "
            f"idx2={idx2}, "
            f"distance_matrix={distance_matrix[idx1, idx2].item():.6f}"
        )
    
        latent1 = latent_vectors[idx1]
        latent2 = latent_vectors[idx2]

        # Interpolate between the two latent vectors and generate the corresponding outputs 
        interpolated_latents, generated_outputs, _ = self._interpolate_latent_points( generator,
                                                                                      latent1, latent2, 
                                                                                      num_interpolations,
                                                                                      extrapolate )
        label = f"Interpolation between models ({idx1} -> {idx2})"
        z_full = self._latent_to_full_image(interpolated_latents) 
        fig = self._plot_zy( interpolated_latents, z_full, generated_outputs, label)
        mlflow_client.log_figure( run.info.run_id, fig, strategy + extrapolate_label + "_latent_space_smoothness.png")
    
        log.info( f"Generated {num_interpolations} " f"interpolated latent samples " f"between {idx1} and {idx2}.")

        return latent2 - latent1

    def eval_latent_space_smoothness( self, mlflow_client, run, generator, encoder, num_interpolations=10, extrapolate=False):
        """
        Evaluate latent-space smoothness.
        """

        latent_vectors, distance_matrix = self._build_latent_vectors_and_cosine_distance(encoder=encoder, X=self.cropped_x_test)

        p_diff = dict()

        strategies=( "max", "min", "random")  # Choose the strategy for selecting latent points: "min", "max", or "random" 
        for s in strategies:
            p_diff[s] = self._eval_by_strategy(mlflow_client, run, generator, num_interpolations, s, latent_vectors, distance_matrix, extrapolate)

        return p_diff

    def eval_generator(self, mlflow_client=None, run=None, generator=None, encoder=None):

        # latent = torch.as_tensor( latent_value, dtype=torch.float32) # for inversion

        # Generate latent space samples
        #n = self.g.gvars.get_raw_test_data().get_current_data().get_X().shape[0]    
        #z = torch.randn(n,512,21,25).to(self.device) # (batch, channels, height, width)

        # Predict 
        #y_hat = self.predict(generator, encoder.predict(self.cropped_x_test))
        y_hat = self.predict(generator, self.predict(encoder, self.cropped_x_test))

        # Evaluate
        mse, mae, ssim, ssim_qmap = self.evaluation(self.cropped_y_test, y_hat)
        # Log results and parameters in mlflow
        self.log_evaluation_results(
                mlflow_client=mlflow_client, run=run, 
                mae=mae, mse=mse, ssim=ssim, ssim_qmap=ssim_qmap, 
                y=self.cropped_y_test, y_hat=y_hat, label="gen"
            )

        return mse, mae, ssim, ssim_qmap

    def eval_autoencoder(self, mlflow_client=None, run=None, model=None):

        # Predict 
        y_hat = self.predict(model, self.cropped_x_test)
        # Evaluate
        mse, mae, ssim, ssim_qmap = self.evaluation(self.cropped_y_test, y_hat)
        # Log results 
        self.log_evaluation_results(
                mlflow_client=mlflow_client, run=run, 
                mae=mae, mse=mse, ssim=ssim, ssim_qmap=ssim_qmap, 
                y=self.cropped_y_test, y_hat=y_hat, label="ae"
            )
        return mse, mae, ssim, ssim_qmap

    def predict(self, skorch_model, X):
        log.info(f"skorch_mode xxx = {type(skorch_model)}")
        skorch_model.module_.eval() # it might not be necessary, because skorch probably does it internally in predict
        return skorch_model.predict(X)

    def evaluation(self, Y, Y_hat):
        mae, _  = self.mae(Y, Y_hat)
        mse, _  = self.mse(Y, Y_hat)
        ssim, ssim_qmap = self.ssim(Y, Y_hat)
        if self.g.verbose:
            log.info(f"MSE over test set is: mean={mse.mean()}, std={mse.std()}, min={mse.min()}, max={mse.max()}")
            log.info(f"MAE over test set is: mean={mae.mean()}, std={mae.std()}, min={mae.min()}, max={mae.max()}")
            log.info(f"SSIM over test set is: mean={ssim.mean()}, std={ssim.std()}, min={ssim.min()}, max={ssim.max()}")
        return mse, mae, ssim, ssim_qmap   

    def log_params(self, mlflow_client, run, skorch_model, lr_scheduler=None, patience=None):   

        model_size=sum(p.numel() for p in skorch_model.module_.parameters())

        self.log_parameters(mlflow_client=mlflow_client, run=run, 
                params=self.task.get_item(0), 
                model_size=model_size, lr_scheduler=lr_scheduler,
                patience=patience, seed=self.g.gvars.get_seed(),
                test_perc=self.g.gvars.get_test_perc(), train_indexes=self.g.gvars.get_train_indexes(), test_indexes=self.g.gvars.get_test_indexes()
            )

    #def set_torch_device(self, device):
    #    self.device = torch.device(device)
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()

    def create_model(self, cb=None, device=None, grid=None):

        if self.g.verbose_level > 0:
            n_channels = self.model_data_size['nchannels']
            log.info(f"N_CHANNELS = {n_channels}")
            log.info(f"is_transformer = {self.g.gvars.is_transformer()}")
            log.info(f"task device = {self.task.get_device()}")
            log.info(f"device = {device}")
            log.info(f"device type = {type(device)}")
            log.info(f"cuda current device = {torch.cuda.current_device()}")
            log.info(f"type of grid = {type(grid)}")
            log.info(f"grid = {grid}")

        autoencoder = ScoredSkorchModel(
            module=EncoderDecoder,
            device=device,
            callbacks=cb,
            seed=self.g.gvars.get_seed(),
            **grid["inference_grid"]["skorch_model_params"][0]
        )

        autoencoder.initialize()

        if self.g.verbose:
            input_size = ( 
                1, 
                self.model_data_size['nchannels'],
                self.model_data_size['nz_cut'], 
                self.model_data_size['nx_cut'],
            )
            print(autoencoder.module_.summary(input_size=input_size))

        autoencoder.load_params(
            f_params=grid["inference_grid"]["weights_files"][0],
#            f_optimizer=grid["optimizer_file"],
#            f_criterion=grid["criterion_file"],
        )

        encoder = SkorchEncoderModelBuilder().build(autoencoder=autoencoder.module_, device=device)
        encoder.initialize()    

        decoder = SkorchGeneratorModelBuilder().build(autoencoder=autoencoder.module_, device=device)
        decoder.initialize()

        return autoencoder, encoder, decoder



class InversionTaskProcessing(InferenceTaskProcessing):
    def __init__(self, task_processing_globals=None, task=None):
        super().__init__(task_processing_globals=task_processing_globals, task=task)
        self.generator = None
        self.encoder = None

    def run(self):
        assert str(self.device) == self.task.get_device(), f"Logical internal error"
        utl.set_seed(self.g.gvars.get_seed())

        # find model architecture compatible sizes
        self.model_data_size = self.get_model_data_size()
        if self.g.verbose:
            log.info(f"model_data_size: {self.model_data_size}")
            log.info(f"[ START ] Refit using taks id {self.task.get_id()} with {self.task.get_n_items()} hyper-parameters instances.")

        # set the target device
        if torch.cuda.is_available():
            print (f"-----------------------")
            print (f"cuda current device : {torch.cuda.current_device()}")
            print (f"self.device : {self.device}")
            print (f"-----------------------")
            torch.cuda.device(self.device)
            print (f"[ thread {self.task.get_id()} cuda current device : {torch.cuda.current_device()}")
            torch.cuda.empty_cache()
            self.stream = torch.cuda.Stream(device=self.device)
        #self.set_torch_device(self.task.get_device())
        
        with torch.cuda.stream(self.stream):

            # deal with mlflow client and run
            mlflow_client, run = MLFlowUtils.create_client_and_get_run(
                mlflow_=self.g.mlflow, 
                experiment_id=self.g.gvars.get_experiment_tracking_id(),
                execution_id=self.g.gvars.get_execution_id(), 
                execution_name=self.g.gvars.get_execution_name()
            )

            _, refit_cbs, lr_scheduler, patience = self.create_callbacks(
                mlflow_client=mlflow_client, 
                run=run,
                is_one_step_training=True, is_inversion=True,
            )
            log.info(f"refit_cbs = {refit_cbs}")


            # Create models
            print (f"-----------------------")
            print (f"[ thread {self.task.get_id()} cuda current device : {torch.cuda.current_device()}")
            log.debug("Calling create_model with device and task")
            log.line()
            log.info(f"grid = {self.task.get_item(0)['inference_grid']['skorch_model_params']}")
            log.line()
            grid =self.task.get_item(0)
            self.model, self.encoder, self.generator, self.inversion = self.create_model(device=self.device, grid=grid, cb=refit_cbs)

            # Prepare datasets
            self.cropped_x, self.cropped_y, self.cropped_x_test, self.cropped_y_test = self.crop_data(device=None)


            latent_idx = self.g.gvars.get_latent_init_idx()

            models_to_invert = [1, 10, 20]
            inverted_models = dict()

            for idx in models_to_invert: 

                self.inversion.initialize() # construct a new inversion object

                if grid['latent_init'] == "trained":
                    self.inversion.module_.set_latent(self.predict(self.encoder,self.cropped_x[latent_idx:latent_idx+1]))
    
                x_test = self.cropped_x_test[idx:idx+1]
                y_test = self.cropped_y_test[idx:idx+1]
                self.inversion.partial_fit(x_test, y_test) # it is important to not call inversion.initialize() again after set_latent()
    
                # Generate from inverted latent
                y_hat = self.predict(self.generator, self.inversion.module_.latent)
    
                # use mse (loss) and ssim (index) metrics
                mse, _  = self.mse(y_test, y_hat)
                mae, _  = self.mae(y_test, y_hat)
                ssim, ssim_qmap = self.ssim(y_test, y_hat)
                if self.g.verbose:
                    log.info(f"\t[ {idx} ] MSE mean over test set is: {mse.mean()}")
                    log.info(f"\t[ {idx} ] MAE mean over test set is: {mae.mean()}")
                    log.info(f"\t[ {idx} ] SSIM mean over test set is: {ssim.mean()}")

                inverted_models[idx] = { "y_hat": y_hat, "y": y_test, "mse" : mse, "mae": mae, "ssim": ssim, "ssim_qmap": ssim_qmap}
                
                # Log training history
                latent_h, y_hat_h, loss_h, epoch_h = self._collect_history(self.g.gvars.get_training_history())
                self.log_training_history(mlflow_client, run, idx, latent_h, y_hat_h, loss_h, epoch_h)

            results = self._collect_results(inverted_models) 

            mse = results["mse"]
            mae = results["mae"]
            ssim = results["ssim"]
            ssim_qmap = results["ssim_qmap"]
            y = results["y"]
            y_hat = results["y_hat"]

            log.info(f"y.shape = {y.shape}")

            self.log_evaluation_results(
                mlflow_client=mlflow_client, run=run, label="inversion",
                mae=mae, mse=mse, ssim=ssim, ssim_qmap=ssim_qmap, 
                y=y, y_hat=y_hat
            )

            torch_model = self.model.module_
            model_size=sum(p.numel() for p in torch_model.parameters())
            self.log_parameters(mlflow_client=mlflow_client, run=run, 
                params=self.task.get_item(0), 
                model_size=model_size, lr_scheduler=lr_scheduler,
                patience=patience, seed=self.g.gvars.get_seed(),
                test_perc=self.g.gvars.get_test_perc(), train_indexes=self.g.gvars.get_train_indexes(), test_indexes=self.g.gvars.get_test_indexes()
            )


            # Placeholder for actual evaluation results   
            #mse, mae, ssim = np.random.rand(3), np.random.rand(3), np.random.rand(3)

        if self.g.verbose:
            log.info(f"[ END ] Inversion finished.")

        return mse, mae, ssim

    def log_training_history(self, mlflow_client, run, idx, z, y, loss, epoch):
        filename = f"{idx}_inversion_history"
        label = f"Inverions history on model {idx}"

        z_full = self._latent_to_full_image( z )
        fig = self._plot_zy( z, z_full, y, label, global_palette=False)
        mlflow_client.log_figure( run.info.run_id, fig, f"{filename}.png")


    def create_model(self, cb=None, device=None, grid=None):

        autoencoder, encoder, generator = super().create_model(cb=None, device=device, grid=grid) 

        inversion = SkorchGeneratorInversionBuilder().build( 
            generator=generator.module_, latent_shape=(1, 512, 21, 25), latent_init=grid['latent_init'], latent_value=None, 
            seed=self.g.gvars.get_seed(), device=device, max_epochs=grid['max_epochs'], lr=grid['lr'], callbacks=cb
        )
        inversion.initialize()    

        return autoencoder, encoder, generator, inversion

    def _collect_results(self, inverted_models):
    
        keys = sorted(inverted_models)
    
        scalar_metrics = ["mse", "mae", "ssim"]
        tensor_metrics = ["ssim_qmap", "y", "y_hat"]
    
        results = {"idx": np.array(keys)}
    
        for metric in scalar_metrics:
            results[metric] = np.array( [inverted_models[idx][metric] for idx in keys])
    
        for metric in tensor_metrics:
            results[metric] = np.concatenate( [inverted_models[idx][metric] for idx in keys])
    
        return results

    def _collect_history(self, history):

        latent_h = torch.cat( [ l for l in history["latent"]], dim=0 )
        y_hat_h  = torch.cat( [ m for m in history["y_hat"]], dim=0 )
        loss_h = torch.as_tensor( history["loss"] )
        epoch_h = torch.as_tensor( history["epoch"] )
    
        return latent_h, y_hat_h, loss_h, epoch_h




