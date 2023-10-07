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


from keras.layers import Flatten, LSTM, Dense, Conv2D, Conv3D, GlobalAveragePooling1D, Dropout, MaxPooling2D

from tensorflow import keras
from tensorflow.keras import layers

from data_prep import data_loader_v1
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
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
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
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
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
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
        n_classes=4
    ):
    inputs = keras.Input(shape=input_shape)
    dense_dim = 8
    embed_dim = input_shape[1]
    
    x = PositionalEmbedding(
        input_shape[0], embed_dim, name="frame_position_embedding"
    )(inputs)
    
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

    x = layers.GlobalMaxPooling1D()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

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

    data = data_loader_v1(model_name, val=val, scale=True)
    if len(data) == 3:
        train, val, test = data
        val_x, val_y = val
    else:
        train, test = data
    train_x, train_y = train
    test_x, test_y = test
    
    total = train_y.shape[0]
    
    print ("train stats: ")
    print (train_x.shape, train_y.shape)
    
    if val:
        print ("val stats: ")
        print (val_x.shape, val_y.shape)
    
    print ("test stats: ")
    print (test_x.shape, test_y.shape)
    
    # class weight strategy
    class_weight = {
        k: (1 / train_y[train_y==k].shape[0]) * (total / 4) for k in np.unique(train_y)
    }
    
    
    BATCH_SIZE = 32

    input_shape = train_x.shape[1:]
        
    model = build_model(
        input_shape,
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
        val_x = test_x
        val_y = test_y
    model.fit(train_x, train_y, 
              validation_data=(val_x, val_y), 
              epochs=200, 
              callbacks=callbacks)
#               class_weight=class_weight)
    
    model.load_weights(f'checkpoints/{get_best_weights()}')
    print ("Evaluating on train set: ")
    model.evaluate(train_x, train_y)

    print ("Evaluating on valid set: ")
    model.evaluate(val_x, val_y)

    print ("Evaluating on test set: ")
    model.evaluate(test_x, test_y)

#     y_pred_train = np.argmax(model.predict(train_x), axis=1)
    y_pred_val = np.argmax(model.predict(val_x), axis=1)
    y_pred_test = np.argmax(model.predict(test_x), axis=1)

    
#     print ("Classification report (train): ")
#     print(classification_report(train_y, y_pred_train))
    

    print ("Classification report (val): ")
    print(classification_report(val_y, y_pred_val))
   

    print ("Classification report (test): ")
    print(classification_report(test_y, y_pred_test))
    
# import sys
import shutil, os
if __name__ == '__main__':
    shutil.rmtree('checkpoints/')
    os.mkdir('checkpoints')
    train(config.MARLIN, val=False)

