
"""@package ml_models_utils.py
Classes with utility functions for machine learning pytorch/skorch models.
"""
import torch
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from skorch.callbacks import Callback, Checkpoint, EarlyStopping, LoadInitState, LRScheduler

from src.utl.logging import Log as log
from src.experiment_tracking.mlflow_skorch_callbacks import TwoPhaseTraining, PrintLRScheduler, \
     MonitoredGradClip, CustomGradientClipping, PredictionHistory           


class MLModelsUtils:

    def get_skorch_two_step_training_callback(nmb_of_epochs=500, step1_perc=0.5, warmup_perc=0.25, finetune_lr=1e-5):
        return TwoPhaseTraining(
            nmb_of_epochs=nmb_of_epochs, 
            step1_perc=step1_perc, 
            warmup_perc=warmup_perc, 
            finetune_lr=finetune_lr
        )

    def get_skorch_inversion_history_callback(history_dict=None, print_every=100):
        return PredictionHistory(history_dict=history_dict, print_every=print_every)

    def get_skorch_monitor_gradient_clipping_callback(clip_value=None, print_every=5):
        return MonitoredGradClip(clip_value=clip_value, print_every=print_every)

    def get_skorch_gradient_clipping_callback(clip_value=None):
        return CustomGradientClipping(clip_value=clip_value)

    def get_skorch_loadInitState_callback(checkpoint_cb):
        return LoadInitState(checkpoint_cb)

    def get_skorch_checkpoint_callback(fn_prefix='', dirname='', load_best=True):
        return Checkpoint(
                            monitor='valid_loss_best',
                            f_params='params.pt',
                            f_optimizer='optimizer.pt',
                            f_criterion='criterion.pt',
                            f_history='history.json',
                            fn_prefix=fn_prefix,
                            dirname=dirname,
                            event_name='event_cp',
                            load_best=load_best
        )

    def get_skorch_early_stopping_callback(patience=20):
        return EarlyStopping(
                            monitor='valid_loss', # explicitly for legibility
                            patience=patience
        )

    def get_skorch_lr_scheduler_lambda_callback(decay=0.0002):
        lr_lambda = lambda epoch: 1 / (1 + decay * epoch)
        return LRScheduler(policy=LambdaLR, lr_lambda=lr_lambda), "decay" 

    def get_skorch_lr_scheduler_exponential_callback(gamma=0.95):
        return LRScheduler(policy=ExponentialLR, gamma=gamma), "exponential"

    def get_skorch_lr_scheduler_cosine_annealing_warm_restarts_callback(T_0=10, T_mult=3, eta_min=1e-6):
        return LRScheduler(policy=CosineAnnealingWarmRestarts, T_0=T_0, T_mult=T_mult, eta_min=eta_min), "cosine_annealing_wr"

    def get_skorch_lr_scheduler_cosine_annealing_callback(T_max=150):
        return LRScheduler(policy=CosineAnnealingLR, T_max=T_max), "cosine_annealing"

    def get_skorch_print_lr_scheduler_callback():
        return PrintLRScheduler()









