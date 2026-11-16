"""@package global_vars.py
Module with global variables.
"""

from pathlib import Path

import random
import numpy as np
import torch
import ml_collections


from src.utl.logging import Log as log
from src.utl.classes import Base
from src.utl.classes import Utils as utl
from src.utl.classes import ProjectDirs, BuildDirectories

from src.data.data_builder import BuildDeepTomographyDataAndModels
from src.data.data import ModelIdx

from src.experiment_tracking.mlflow_utils import MLFlowUtils as mlflow_utils

#from src.utl.classes import Utils as utl
#from src.utl.classes import Graph as graph

def load_data_and_models(data=None, data_type=None, data_resolution=None, data_max_samples=None, test_perc=None, seed=None):

    if data == 'shallow':
        data_basedir = "/gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data/data/"
        models_mask=[False] * ModelIdx.MAX_MODEL_IDX
        models_mask[ ModelIdx.TRUE_MODEL_IDX] = True
        models_mask[ ModelIdx.HIGHRES_MODEL_IDX] = True
# NEWER
    else:
        data_basedir = "/gemini/tg/ntbs/doutorado/5semestre/estudo.orientado/deeptomograph_data/data_new_and_bigger/"
        models_mask=[True] * ModelIdx.MAX_MODEL_IDX

    dirs = BuildDirectories(data=data_type, basedir=data_basedir).build()

    ##############################
    # create train and test data
    ##############################
    
    data_and_models_builder = BuildDeepTomographyDataAndModels(
        dirs=dirs, models_mask=models_mask, seed=seed,
        test_perc=test_perc, max_samples=data_max_samples, verbose=True)
    train_data, test_data, train_models, test_models, train_indexes, test_indexes = data_and_models_builder.build(True)


    # setting data resolution
    train_data.set_resolution(data_resolution)
    if test_data is not None:
        test_data.set_resolution(data_resolution)

    newer_data = data_and_models_builder.is_newer_data()

    return data_basedir, dirs, \
        train_data, test_data, \
        train_models, test_models, \
        train_indexes, test_indexes, \
        newer_data



class BaseGlobalVars(Base):
    """!
    Contains globals variables.
    """

    def __init__(
        self,
        verbose_level=None, # means no verbose
        mlflow_uri=None,
        mlflow_experiment=None,
        mlflow_experiment_id=None,
        sep="__",
        name='BaseGlobalVars'
    ):
        super().__init__(name=name)
        self.verbose_level = verbose_level
        self.verbose = (verbose_level > 0)
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_experiment_id = mlflow_experiment_id
        self.sep = sep
        self.name = name

    def is_verbose(self):
        return self.verbose

    def get_verbose_level(self):
        return self.verbose_level
    
    def get_experiment_tracking_uri(self):
        return self.mlflow_uri

    def get_experiment_tracking_name(self):
        return self.mlflow_experiment
    
    def get_experiment_tracking_id(self):
        return self.mlflow_experiment_id

    def set_experiment_tracking_id(self, id):
        self.mlflow_experiment_id = id
    
class LoadDataBaseGlobalVars(BaseGlobalVars):
    """
    Contains globals variables associated to the loading of data and models.    
    """
    def __init__(
        self,
        *args, 
        checkpoint_basedir=None,
        data="deep", #"shallow"
        data_resolution='high', #'med', 'low'
        model_basedir=None,   # evaluation-only
        seed=-1,
        data_max_samples=-1,
        test_perc=0.1,
        label_id=None,
        execution_id=None,
        device='cuda:all',
        **kwargs,
    ):
        kwargs['name'] = 'LoadDataBaseGlobalVars'
        super().__init__(*args, **kwargs)
#        super().__init__(name=name, verbose_level=verbose_level, mlflow_uri=mlflow_uri, 
#            mlflow_experiment=mlflow_experiment, mlflow_experiment_id=mlflow_experiment_id,
#            sep=sep
#            )

        self.checkpoint_basedir = checkpoint_basedir
        self.data = data
        self.data_resolution = data_resolution
        self.data_max_samples = data_max_samples
        self.model_basedir = model_basedir
        self.execution_id = execution_id
        self.execution_name = None
        self.label_id = label_id
        self.test_perc = test_perc
        self.device = device
        self.timestamp = utl.get_timestamp() 
        self.restart = None

        ##############################
        # managing reproducibility 
        ##############################
        self.seed = None if seed < 0 else seed
        if self.seed is not None:
            utl.set_seed(self.seed)

        if self.execution_id is not None:
            assert self.checkpoint_basedir is not None, f"If it is restart, attribute checkpoint_dir must be provided."
            self.restart = True
        else:
            self.restart = False

    def is_restart(self):
        return self.restart
    
    def get_data_type(self):
        return self.data
    
    def get_data_resolution(self):
        return self.data_resolution
    
    def get_data_max_samples(self):
        return self.data_max_samples
    
    def get_checkpoint_basedir(self):
        return self.checkpoint_basedir
    
    def get_data_basedir(self):
        return self.data_basedir
    
    def get_model_basedir(self):
        return self.model_basedir
    
    def get_dirs(self):
        return self.dirs

    def get_execution_id(self):
        return self.execution_id
    
    def set_execution_id(self, id):
        self.execution_id = id
    
    def get_execution_name(self):
        return self.execution_name
    
    def set_execution_name(self, name):
        self.execution_name = name

    def get_timestamp(self):
        return self.timestamp
    
    def get_test_perc(self):
        return self.test_perc
    
    def get_label_id(self):
        return self.label_id
    
    def get_seed(self):
        return self.seed

    def get_device(self):
        return self.device

class LoadDataGlobalVars(LoadDataBaseGlobalVars):
    """!
    Contains globals variables with the data and models loaded.
    """
    def __init__(
        self,
        *args, 
        **kwargs,
    ):
        kwargs['name'] = 'LoadDataGlobalVars'
        super().__init__(*args, **kwargs)

        (self.data_basedir, self.dirs,         
        self.train_data, self.test_data,      
        self.train_models, self.test_models,  
        self.train_indexes, self.test_indexes, 
        self.newer_data) = load_data_and_models(
            data=self.data, 
            data_type=self.data, 
            data_resolution=self.data_resolution, 
            data_max_samples=self.data_max_samples, 
            test_perc=self.test_perc, 
            seed=self.seed          
        )


    def print(self):
        if (self.verbose):
            log.info(f"Project info: {self.dirs}")
            log.info(f"Is newer data? {self.is_newer_data}")
            log.info(f"Experiment tracking URI: {self.mlflow_uri}")
            log.info(f"Experiment tracking name: {self.mlflow_experiment}")
            log.info(f"Experiment tracking id: {self.mlflow_experiment_id}")
            log.info(f"Seed: {self.seed}")
            log.info(f"Max samples: {self.data_max_samples}")

    def is_newer_data(self):
        return self.newer_data
    
    def get_dirs(self):
        return self.dirs
    
    def get_raw_train_data(self):
        return self.train_data

    def get_raw_train_models(self):
        return self.train_models

    def get_raw_test_data(self):
        return self.test_data

    def get_raw_test_models(self):
        return self.test_models

    def get_train_indexes(self):
        return self.train_indexes

    def get_test_indexes(self):
        return self.test_indexes


class GlobalVars(LoadDataGlobalVars):
    """!
    Contains globals variables.
    """
    def __init__(
        self,
        *args,
        unembed_type='paper',
        conv_filter_size=(13,13),
        conv_dilation_size=(1,1),
        transposed_conv_filter_size=(2,2),
        max_epochs=130,
        cv=5,
        one_step_training=True, 
        is_vae=False,
        use_transformer=False,
        **kwargs
    ):
        kwargs['name'] = 'GlobalVars'
        super().__init__(*args, **kwargs)
#        super().__init__(name=name, verbose_level=verbose_level, mlflow_uri=mlflow_uri, 
#            mlflow_experiment=mlflow_experiment, mlflow_experiment_id=mlflow_experiment_id,
#            sep=sep, checkpoint_basedir=checkpoint_basedir, data=data, data_resolution=data_resolution,
#            model_basedir=model_basedir, seed=seed, data_max_samples=data_max_samples,
#            test_perc=test_perc, label_id=label_id, execution_id=execution_id
#            )

        self.unembed_type = unembed_type
        self.conv_filter_size = conv_filter_size
        self.conv_dilation_size = conv_dilation_size
        self.transposed_conv_filter_size = transposed_conv_filter_size
        self.max_epochs = max_epochs
        self.cv = cv
        self.one_step_training = one_step_training
        self.vae = is_vae
        self.transformer = use_transformer


    def get_conv_filter_size(self):
        return self.conv_filter_size

    def get_conv_dilation_size(self):
        return self.conv_dilation_size

    def get_transposed_conv_filter_size(self):
        return self.transposed_conv_filter_size

    def get_unembed_type(self):
        return self.unembed_type

    def get_max_epochs(self):
        return self.max_epochs

    def get_execution_ids(self):
        return self.execution_id.split(',')
    
    def get_cv(self):
        return self.cv
    
    def is_one_step_training(self):
        return self.one_step_training

    def is_vae(self):
        return self.vae

    def is_transformer(self):
        return self.transformer


class RefitGlobalVars(GlobalVars):
    """!
    Implements the command line interface for refit -- final training -- and evaluation.
    """
    def __init__(self,
            *args, 
            model_path=None, refit=None, att_num_layers=None, gs_best_run_id=None,
            gs_best_run_experiment_tracking_name=None, 
            **kwargs
        ):
        kwargs['name'] = 'RefitGlobalVars'
        super().__init__(*args, **kwargs)
        self.refit = refit
        self.model_path = model_path
        self.att_num_layers = att_num_layers
        self.gs_best_run_id = gs_best_run_id
        self.gs_best_run_experiment_tracking_name = gs_best_run_experiment_tracking_name
        self.gs_best_run_experiment_tracking_id = None #MLFlowUtils.get_experiment_id_by_experiment_name(self.gs_best_run_experiment_tracking_name)
        assert self.refit is not None, f"refit parameter must be set!"
        assert self.model_path is not None, f"model_path parameter must be set!"

    def get_best_run_experiment_tracking_name(self):
        return self.gs_best_run_experiment_tracking_name
    
    def get_best_run_experiment_tracking_id(self):
        return self.gs_best_run_experiment_tracking_id
    
    def set_best_run_experiment_tracking_id(self, id):
        self.gs_best_run_experiment_tracking_id = id

    def get_best_run_id(self):
        return self.gs_best_run_id
    
    def get_model_path(self):
        return self.model_path
    
    def get_att_num_layers(self):
        return self.att_num_layers
    
    def is_refit(self):
        return self.refit

    
class InferenceGlobalVars(LoadDataBaseGlobalVars):
    """!
    Implements the command line interface for inference.
    """
    def __init__(self,
            *args, 
            model_path=None,  
            refit_run_id=None,
            refit_experiment_tracking_name=None,
            use_transformer=None,
            max_epochs=None,
            latent_init=None,
            **kwargs
        ):
        kwargs['name'] = 'InferenceGlobalVars'
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.args = args
        self.refit_run_id = refit_run_id
        self.refit_experiment_tracking_name = refit_experiment_tracking_name
        self.model_path = model_path
        self.transformer = use_transformer
        self.max_epochs = max_epochs
        self.latent_init = latent_init

        # Attributes that will be set after mlflow initialization and reading the refit run attributes
        self.refit_run_experiment_tracking_id = None
        self.refit_run_name = None
        self.conv_filter_size = None
        self.conv_dilation_size = None
        self.transposed_conv_filter_size = None
        self.attention_config = None
        self.model_size = None

        # Attributes that will be set after loading the data
        self.data_basedir = None
        self.dirs = None
        self.train_data = None
        self.test_data = None
        self.train_models = None
        self.test_models = None
        self.train_indexes = None
        self.test_indexes = None
        self.newer_data = None
        
        assert self.model_path is not None, f"model_path parameter must be set!"


#        #def get_param_from_run(run=None, key=None):
#        attr = self.read_refit_run_id_attributes(
#            experiment_tracking_uri=args.mlfow_uri,
#            experiment_id=refit_experiment_tracking_name,
#            run_id=refit_run_id,
#        )
#        self.seed = attr.get('seed', None)
#        checking_seed = utl.get_seed_from_name(self.refit_experiment_tracking_name)
#        assert self.seed == checking_seed, f"Seed mismatch between refit run and experiment tracking!"


    def load_data(self):
        (self.data_basedir, self.dirs,
        self.train_data, self.test_data, 
        self.train_models, self.test_models, 
        self.train_indexes, self.test_indexes,
        self.newer_data) = load_data_and_models(
            data=self.data, 
            data_type=self.data, 
            data_resolution=self.data_resolution, 
            data_max_samples=self.data_max_samples, 
            test_perc=self.test_perc, 
            seed=self.seed          
        )

    def is_transformer(self):
        return self.transformer

    def get_model_path(self):
        return self.model_path
    
    def get_refit_run_id(self):
        return self.refit_run_id

    def get_refit_run_name(self):
        return self.refit_run_name

    def get_refit_experiment_tracking_name(self):
        return self.refit_experiment_tracking_name

    def get_conv_filter_size(self):
        return self.conv_filter_size

    def get_conv_dilation_size(self):
        return self.conv_dilation_size

    def get_transposed_conv_filter_size(self):
        return self.transposed_conv_filter_size

    def get_attention_config(self):
        return self.attention_config

    def get_model_size(self):
        return self.model_size

    def is_newer_data(self):
        return self.newer_data
    
    def get_dirs(self):
        return self.dirs
    
    def get_raw_train_data(self):
        return self.train_data

    def get_raw_train_models(self):
        return self.train_models

    def get_raw_test_data(self):
        return self.test_data

    def get_raw_test_models(self):
        return self.test_models

    def get_train_indexes(self):
        return self.train_indexes

    def get_test_indexes(self):
        return self.test_indexes

    def set_refit_run_experiment_tracking_id(self, id):
        self.refit_run_experiment_tracking_id = id

    def set_refit_run_name(self, name):
        self.refit_run_name = name 

    def set_attributes_from_refit_run_attributes(self, run_attributes=None):
        assert run_attributes is not None, f"run_attributes parameter must be provided!"

        seed = None
        seed = eval(run_attributes.get('_x_seed', seed))
        assert seed == self.seed, f"Seed mismatch between refit run anda command line arguments!"

        self.conv_filter_size = eval(run_attributes.get('module__conv_kernel_size', self.conv_filter_size))

        self.transposed_conv_filter_size = eval(run_attributes.get('module__transposed_conv_kernel_size', self.conv_filter_size))

        self.conv_dilation_size = eval(run_attributes.get('module__conv_dilation_size', self.conv_dilation_size))

        self.model_size = eval(run_attributes.get('model_size', self.model_size))

        transformer = eval(run_attributes.get('module__use_transformers', "False"))
        assert transformer == self.transformer, f"Test percentage mismatch between refit run and command line arguments!"

        test_perc = eval(run_attributes.get('_x_test_perc', self.transformer))
        assert test_perc == self.test_perc, f"Test percentage mismatch between refit run and command line arguments!"

        self.attention_config = ml_collections.ConfigDict(mlflow_utils.string_to_dict(run_attributes['module__attention_config']))
        self.attention_config.patches = ml_collections.ConfigDict({'size': (1, 1)})
    


class InversionGlobalVars(InferenceGlobalVars):
    """!
    Implements the command line inversion.
    """
    def __init__(self,
            *args, 
            max_epochs=None,
            latent_init=None,
            latent_init_idx=None,
            lr=None,
            **kwargs
        ):
        kwargs['name'] = 'InversionGlobalVars'
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.args = args
        self.max_epochs = max_epochs
        self.latent_init = latent_init
        self.latent_init_idx = latent_init_idx
        self.lr = lr

        self.training_history = dict()

    def get_latent_init(self):
        return self.latent_init

    def get_max_epochs(self):
        return self.max_epochs

    def get_lr(self):
        return self.lr

    def get_latent_init_idx(self):
        return self.latent_init_idx

    def get_training_history(self):
        return self.training_history


