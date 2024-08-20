
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                            MaxPooling2D, Concatenate, UpSampling2D, BatchNormalization, 
                            ReLU, Dense, Flatten, Add, Activation, 
                            UpSampling2D, multiply, Concatenate, Dropout)
from keras import optimizers as opt

#from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, Dense, Flatten, Add, Activation
#from tensorflow.keras.layers import UpSampling2D, multiply, Concatenate, Dropout


def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)

def jaccard_coef(y_true, y_pred, smooth=0.0):

    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def Attention_block(filters, x, g):
    
    g = UpSampling2D(size=2)(g)
    
    wg = Conv2D(filters, kernel_size=1, strides=1, padding="valid", use_bias=True)(g)
    wg = BatchNormalization()(wg)
    
    wx = Conv2D(filters, kernel_size=1, strides=1, padding="valid", use_bias=True)(x)
    wx = BatchNormalization()(wx)
    
    psi = Add()([wx, wg])
    psi = Activation('relu')(psi)
    
    psi = Conv2D(1, kernel_size=1, strides=1, padding="valid", use_bias=True)(psi)
    psi = BatchNormalization()(psi)    
    psi = Activation("sigmoid")(psi)    
    
    return multiply([psi,x])

def create_Att_MFP_unet_model2D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5,5),
                        deconvolution_kernel_size=(5,5),
                        pool_size=(2,2),
                        strides=(2,2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001):
    """
    Create a 2D Unet model

    Example
    -------
    unet_model = create_Unet_model2D( (100,100,1), 1, 4)
    """
    layers = np.arange(layers)
    number_of_classification_labels = n_labels
    
    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2**(layers[i])

        if i == 0:
            conv = Conv2D(filters=number_of_filters, 
                                kernel_size=convolution_kernel_size,
                                dilation_rate=3,
                                activation='relu',
                                padding='same')(inputs)
        else:
            conv = Conv2D(filters=number_of_filters, 
                                kernel_size=convolution_kernel_size,
                                dilation_rate=3,
                                activation='relu',
                                padding='same')(pool)

        encoding_convolution_layers.append(Conv2D(filters=number_of_filters, 
                                                        kernel_size=convolution_kernel_size,
                                                        dilation_rate=3,
                                                        activation='relu',
                                                        padding='same')(conv))

        if i < len(layers)-1:
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers)-1]
    feature_maps = []
    
    #feature0 = outputs
    feature = Conv2D(filters=lowest_resolution, kernel_size=convolution_kernel_size,
                     dilation_rate=3,
                     activation='relu', padding='same')(outputs)
    feature_maps.append(feature)
    
    for i in range(1,len(layers)):
        number_of_filters = lowest_resolution * 2**(len(layers)-layers[i]-1)
        
        #if i == 0:
         #   att = Attention_block(filters=number_of_filters, 
         #                 x=encoding_convolution_layers[len(layers)-i-1], 
         #                 g=feature0)
        #else:
        att = Attention_block(filters=number_of_filters, 
                        x=encoding_convolution_layers[len(layers)-i-1], 
                        g=feature)
        
        tmp_deconv = Conv2DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     dilation_rate=3,
                                     padding='same')(outputs)
        tmp_deconv_up = UpSampling2D(size=pool_size)(tmp_deconv)
        
        
        #att_up  = UpSampling2D(size=pool_size)(att)
        
        outputs = Concatenate(axis=3)([tmp_deconv_up, att])

        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                         dilation_rate=3,
                        activation='relu', padding='same')(outputs)
        outputs = Conv2D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                         dilation_rate=3,
                        activation='relu', padding='same')(outputs)
        feature = Conv2D(filters=lowest_resolution, kernel_size=convolution_kernel_size,
                         dilation_rate=3,
                        activation='relu', padding='same')(outputs)
        feature_maps.append(feature)

    if mode == 'classification':
        if number_of_classification_labels == 1:
            for i in range(0, len(layers)):
                up_size = 2**(len(layers)-1-i)
                feature_maps[i] = UpSampling2D(size=up_size)(feature_maps[i])
            final_feature_map = Concatenate(axis=3)(feature_maps)
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1),
                             dilation_rate=3,
                            activation='sigmoid')(final_feature_map)
        else:
            outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1),
                             dilation_rate=3,
                            activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=loss_dice_coefficient_error, 
                                optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient, jaccard_coef])
        else:
            unet_model.compile(loss='categorical_crossentropy', 
                                optimizer=opt.Adam(lr=init_lr), metrics=['accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        for i in range(0, len(layers)):
            up_size = 2**(len(layers)-1-i)
            feature_maps[i] = UpSampling2D(size=up_size)(feature_maps[i])
        final_feature_map = Concatenate(axis=3)(feature_maps)
        
        outputs = Conv2D(filters=number_of_classification_labels, kernel_size=(1,1),
                         dilation_rate=3,
                        activation=output_activation)(final_feature_map)
        unet_model = Model(inputs=inputs, outputs=outputs)
        unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr))
    else:
        raise ValueError('mode must be either `classification` or `regression`')

    return unet_model

