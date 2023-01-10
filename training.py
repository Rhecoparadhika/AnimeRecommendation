import os

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from kgcn import KGCN


def train(
        aggregator: str,
        n_epochs: int,
        dim: int,
        n_iter: int,
        batch_size: int,
        l2_weight: float,
        lr: float,
        output_data_dir: str,
        use_tfrecord: bool = False) -> KGCN:

    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train, valid, test = data[4], data[5], data[6]
    adj_entitiy, adj_relation = data[7], data[8]

    model = KGCN(
        dim=dim,
        n_user=n_user,
        n_entity=adj_entitiy.shape[0],
        n_relation=n_relation,
        adj_entity=adj_entitiy,
        adj_relation=adj_relation,
        n_iter=n_iter,
        aggregator_type=aggregator,
        regularizer_weight=l2_weight)

    adam = Adam(learning_rate=lr)
    escb = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    tbcb = TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        embeddings_freq=1)
    model.compile(optimizer=adam, loss='binary_crossentropy')

    if use_tfrecord:
        model.fit(
            train,
            batch_size=batch_size,
            epochs=n_epochs,
            callbacks=[escb, tbcb],
            validation_data=valid,
            verbose=1)
    else:
        model.fit(
            [train[0], train[1]],
            train[2],
            batch_size=batch_size,
            epochs=n_epochs,
            callbacks=[escb, tbcb],
            validation_data=([valid[0], valid[1]], valid[2]),
            verbose=1)

    return model
