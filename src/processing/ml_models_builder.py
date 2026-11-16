
"""@package ml_models_builder.py
Builders for constructing ml models using skorch wrappers to map pytorch models to sklearn models. 
"""

import torch
import torch.nn as nn

#from torchvision import models
#from torch.nn.functional import relu
#from torchinfo import summary

from skorch import NeuralNetRegressor
from skorch import scoring
from skorch import dataset

import src.transunet.vit_seg_configs as vit_config
from src.transunet.vit_seg_modeling import VisionTransformer

from src.utl.logging import Log as log
from src.utl.classes import Utils as utl
from src.processing.ml_models import EncoderDecoder, Generator, Encoder, GeneratorInversion
#from src.data.data import SupervisedData



class ScoredSkorchModel(NeuralNetRegressor):
    def __init__(self, *args, seed=None, **kwargs):
        self.seed = seed
        super().__init__(*args, **kwargs)

    def get_iterator(self, dataset, training=False):
        iterator = super().get_iterator(dataset, training=training)

         
        if training and getattr(iterator, 'shuffle', False):
            # Recria o gerador travado na semente exata a cada época
            g = torch.Generator(device='cuda') # Certifique-se de alinhar o device se necessário
            g.manual_seed(self.seed)
            iterator.generator = g
            
        return iterator

    def score(self, X, y=None):
        return -scoring.loss_scoring(self, X, y)

#    def set_params(self, **params):
#        if 'criterion' in params:
#            self.criterion = params.pop('criterion')
#            self.initialize_criterion()
#        return super().set_params(**params)
#
#    def initialize_criterion(self):
#        self.criterion_ = self.criterion
#        return self

class BaseModelBuilder:
    def __init__(self, verbose=False):
        self.verbose = verbose

class SkorchEncoderDecoderModelBuilder(BaseModelBuilder):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def build(self,
              seed=None, 
              max_epochs=50,
              batch_size=16,
              device='cuda',
              lr=0.0001,
              optimizer=torch.optim.Adam,
              #criterion=nn.SSIM(),
              criterion=nn.MSELoss(),
              callbacks=None,
              iterator_train__shuffle=True, # shuffle train and validation batches
              optimizer__weight_decay=0, # means without L2 regularization
              train_split=dataset.ValidSplit(5),  # using skorch's default (20% validation)

              in_channels=1,
              out_channels=1,
              n_filters_init=32,
              conv_kernel_size=(3,3),
              conv_dilation_size=(1,1),
              transposed_conv_kernel_size=(2,2),
              pool_kernel_size=(2,2), 
              pool_stride=(2,2),
              batch_normalization=False, 
              dropout=False,
              use_transformers=True,
              unembed_type='linear',
              attention_img_size=(21,25),
              attention_in_channels=512,
              attention_hidden_size=1024,
              attention_num_heads=16,
              attention_num_layers=12,
              attention_mlp_dim=4096,
              vis=False,
              verbose=False,
            ):

        # NOTE: For this moment in_channels and hidden_layer must be the same
        attention_config = vit_config.get_attention_regression_config(hidden_size=attention_hidden_size, 
                                                                      num_layers=attention_num_layers, num_heads=attention_num_heads, mlp_dim=attention_mlp_dim)

        if self.verbose: 
            log.debug(attention_config)
            log.info(f"use_transformers = {use_transformers}")
            log.info(f"unembed_type = {unembed_type}")
            log.info(f"weight decay = {optimizer__weight_decay}")
            log.info(f"train_split = {train_split}")

        #return  NeuralNetRegressor(
        return  ScoredSkorchModel(
            seed=seed,
            module=EncoderDecoder,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            lr=lr,
            optimizer=optimizer,
            criterion=criterion,
            callbacks=callbacks,
        ##### new parameters - test
            iterator_train__shuffle=True,
            iterator_train__worker_init_fn=utl.seed_worker,   
            optimizer__weight_decay=optimizer__weight_decay,
            train_split=train_split,
            #train_split=None, if you are using gridsearchcv without early stopping 
            #train_split=[callable], if you are using gridsearchcv with early stopping 
        ##### FIM new parameters - test
            module__in_channels=in_channels,
            module__out_channels=out_channels, 
            module__n_filters_init=n_filters_init,
            module__conv_kernel_size=conv_kernel_size, 
            module__conv_dilation_size=conv_dilation_size, 
            module__transposed_conv_kernel_size=transposed_conv_kernel_size,
            module__pool_kernel_size=pool_kernel_size,
            module__pool_stride=pool_stride,
            module__batch_normalization=batch_normalization,
            module__dropout=dropout,
            module__use_transformers=use_transformers,
            module__unembed_type=unembed_type,
            module__attention_config=attention_config,
            module__attention_img_size=attention_img_size,
            module__attention_in_channels=attention_in_channels,
            module__vis=vis,
            module__verbose=self.verbose,
            verbose=self.verbose,
        )



class SkorchGeneratorModelBuilder(BaseModelBuilder):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def build(self, max_epochs=1000, batch_size=1, device=None, lr=1e-3, optimizer=torch.optim.Adam, criterion=nn.MSELoss(), callbacks=None, autoencoder=None):
        if autoencoder is None:
            raise ValueError("autoencoder must be provided for SkorchGeneratorModelBuilder")

        self.autoencoder = autoencoder

        return  ScoredSkorchModel(
            module=Generator,
            module__autoencoder=self.autoencoder,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            lr=lr,
            optimizer=optimizer,
            criterion=criterion,
            callbacks=callbacks,
            iterator_train__shuffle=False,
            train_split=None,
        )


class SkorchEncoderModelBuilder(BaseModelBuilder):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def build(self, max_epochs=1000, batch_size=1, device=None, lr=1e-3, optimizer=torch.optim.Adam, criterion=nn.MSELoss(), callbacks=None, autoencoder=None):
        if autoencoder is None:
            raise ValueError("autoencoder must be provided for SkorchEncoderModelBuilder")

        self.autoencoder = autoencoder

        return  ScoredSkorchModel(
            module=Encoder,
            module__autoencoder=self.autoencoder,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            lr=lr,
            optimizer=optimizer,
            criterion=criterion,
            callbacks=callbacks,
            iterator_train__shuffle=False,
            train_split=None,
        )


class SkorchGeneratorInversionBuilder(BaseModelBuilder):

    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def build( self, generator, latent_shape, latent_init=None, latent_value=None, seed=None, max_epochs=500,
        device='cuda', lr=1e-2, callbacks=None,
    ):
        if generator is None:
            raise ValueError("generator must be provided for SkorchGeneratorInversionModelBuilder")

        if latent_init is None and latent_value is None:
            raise ValueError("Either latent_init or latent_value must be specified for SkorchGeneratorInversionModelBuilder")

        if latent_init is not None and latent_value is not None:
            raise ValueError("Only one of latent_init or latent_value should be specified for SkorchGeneratorInversionModelBuilder")

        return ScoredSkorchModel(
            module=GeneratorInversion,
            module__generator=generator,
            module__latent_shape=latent_shape,
            module__latent_init=latent_init,
            module__latent_value=latent_value,
            module__verbose=self.verbose,

            seed=seed,
            callbacks=callbacks,
            max_epochs=max_epochs,
            batch_size=1,
            device=device,
            lr=lr,
            optimizer = torch.optim.Adam, 
            criterion=nn.MSELoss(), 
            iterator_train__shuffle=False,
            train_split=None,
            verbose=self.verbose,
        )










