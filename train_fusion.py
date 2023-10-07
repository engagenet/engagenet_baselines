"""Trainer application code

Author - Ximi
License - MIT
export LD_LIBRARY_PATH=/home/surbhi/anaconda3/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_DIR=/usr/lib/cuda
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_DIR}
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()


from keras.layers import Flatten, Concatenate, LSTM, Dense, Conv2D, Conv3D, GlobalAveragePooling1D, Dropout, MaxPooling2D

from tensorflow import keras
from tensorflow.keras import layers

from data_prep import data_loader_v1, data_loader_fusion
import config
import utils

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim
        })
        return config
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), 
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads
        })
        return config
    
def build_user_model(x, input_shape, num_heads, mlp_units, mlp_dropout):
    dense_dim = 8
    embed_dim = input_shape[1]
    
    x = PositionalEmbedding(
        input_shape[0], embed_dim, name="frame_position_embedding_usr"
    )(x)
    
    
#     x = keras.layers.GlobalMaxPooling1D()(x)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    
#     # late fusin
#     for dim in mlp_units:
#         x = layers.Dense(dim, activation="relu")(x)
#         x = layers.Dropout(mlp_dropout)(x)
        
#     x = layers.Dense(4, activation='softmax')(x)
    return x
    
def build_stm_model(x, input_shape, num_heads, mlp_units, mlp_dropout):
    dense_dim = 8
    embed_dim = input_shape[1]
    
    x = PositionalEmbedding(
        input_shape[0], embed_dim, name="frame_position_embedding_stm"
    )(x)
    
  
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    
#     # late fusing
#     for dim in mlp_units:
#         x = layers.Dense(dim, activation="relu")(x)
#         x = layers.Dropout(mlp_dropout)(x)
        
#     x = layers.Dense(4, activation='softmax')(x)
    return x

# def transformer_enc(
def transformer_encoder(inputs, head_size, num_heads, ff_dim=4, dropout=0.3):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    
    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
        input_shape_1,
        input_shape_2,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
        n_classes=4
    ):
    inputs_1 = keras.Input(shape=input_shape_1)
    inputs_2 = keras.Input(shape=input_shape_2)
    dense_dim = 8
    
    user_x = build_user_model(inputs_1, input_shape_1, num_heads, mlp_units, mlp_dropout)
    stm_x = build_stm_model(inputs_2, input_shape_2, num_heads, mlp_units, mlp_dropout)
    
    x = keras.layers.concatenate([user_x, stm_x])
    
    
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    
    
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model([inputs_1, inputs_2], outputs)

    return model


BUFFER_SIZE = 100000
def make_ds(features, labels):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds

def get_best_weights():
    import os
    def get_epoch(x):
        return int(x.split('epoch')[-1].split('-')[0])
    return sorted(os.listdir('./checkpoints/'), key=get_epoch)[-1]

def train(model_name, val=True):

    data= data_loader_fusion(model_name, val)

    
    
    if len(data) == 3:
        train, val, test = data
        val_x, val_y = val
        val_x1, val_x2 = val_x
    else:
        train, test = data
        
    train_x, train_y = train
    train_x1, train_x2 = train_x
    
    test_x, test_y = test
    test_x1, test_x2 = test_x
    
    total = train_y.shape[0]
    
    print ("train stats: ")
    print (train_x1.shape, train_y.shape)
    print (train_x2.shape, train_y.shape)
    
    if val:
        print ("val stats: ")
        print (val_x1.shape, val_y.shape)
        print (val_x2.shape, val_y.shape)
    
    print ("test stats: ")
    print (test_x1.shape, test_y.shape)
    print (test_x2.shape, test_y.shape)
    
    # class weight strategy
    class_weight = {
        k: (1 / train_y[train_y==k].shape[0]) * (total / 4) for k in np.unique(train_y)
    }
    
    
    BATCH_SIZE = 32

    input_shape_1 = train_x1.shape[1:]
    input_shape_2 = train_x2.shape[1:]
        
    model = build_model(
        input_shape_1,
        input_shape_2,
        head_size=256,
        num_heads=8,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.5,
        dropout=0.3,
        n_classes=4
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    
    filepath = "checkpoints/" + model_name + ".epoch{epoch:02d}-acc{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_accuracy',
                             verbose=1, 
                             save_best_only=True,
                             mode='max')
    
    callbacks = [
                 keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True),
                 checkpoint
    ]
    if not val:
        val_x1 = test_x1
        val_x2 = test_x2
        val_y = test_y
    model.fit([train_x1, train_x2], train_y, 
              validation_data=([val_x1, val_x2], val_y), 
              epochs=200, 
              callbacks=callbacks,
              )
    
    model.load_weights(f'checkpoints/{get_best_weights()}')
    print ("Evaluating on train set: ")
    model.evaluate([train_x1, train_x2], train_y)

    print ("Evaluating on valid set: ")
    model.evaluate([val_x1, val_x2], val_y)

    print ("Evaluating on test set: ")
    model.evaluate([test_x1, test_x2], test_y)

#     y_pred_train = np.argmax(model.predict(train_x), axis=1)
#     y_pred_val = np.argmax(model.predict(val_x), axis=1)
#     y_pred_test = np.argmax(model.predict(test_x), axis=1)

    
#     print ("Classification report (train): ")
#     print(classification_report(train_y, y_pred_train))
    

#     print ("Classification report (val): ")
#     print(classification_report(val_y, y_pred_val))
   

#     print ("Classification report (test): ")
#     print(classification_report(test_y, y_pred_test))
    
# import sys
import shutil, os
if __name__ == '__main__':
    shutil.rmtree('checkpoints/')
    os.mkdir('checkpoints')
    train(config.FUSION, val=False)

