"""Sequence models: GRU, LSTM, 1D-CNN and a small Transformer encoder."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="signlang")
class PositionalEmbedding(layers.Layer):
    """Learned positional embedding added to the input sequence."""

    def __init__(self, seq_len, dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len, self.dim = seq_len, dim

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos", shape=(self.seq_len, self.dim), initializer="random_normal", trainable=True
        )

    def call(self, x):
        return x + self.pos

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "dim": self.dim})
        return cfg


def _head(x, n_classes, dropout):
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(n_classes, activation="softmax")(x)


def build_gru(seq_len, n_features, n_classes, dropout=0.3):
    inp = layers.Input((seq_len, n_features))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
    x = layers.GRU(64, dropout=dropout)(x)
    return keras.Model(inp, _head(x, n_classes, dropout), name="gru")


def build_lstm(seq_len, n_features, n_classes, dropout=0.3):
    inp = layers.Input((seq_len, n_features))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=dropout))(x)
    x = layers.LSTM(64, dropout=dropout)(x)
    return keras.Model(inp, _head(x, n_classes, dropout), name="lstm")


def build_cnn(seq_len, n_features, n_classes, dropout=0.3):
    inp = layers.Input((seq_len, n_features))
    x = inp
    for filters in (128, 192, 256):
        x = layers.Conv1D(filters, 5, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(inp, _head(x, n_classes, dropout), name="cnn")


def build_transformer(seq_len, n_features, n_classes, dropout=0.3, dim=128, heads=4, blocks=2):
    inp = layers.Input((seq_len, n_features))
    x = layers.Dense(dim)(inp)
    x = PositionalEmbedding(seq_len, dim)(x)
    for _ in range(blocks):
        h = layers.LayerNormalization()(x)
        h = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=dropout)(h, h)
        x = layers.Add()([x, h])
        h = layers.LayerNormalization()(x)
        h = layers.Dense(dim * 2, activation="gelu")(h)
        h = layers.Dropout(dropout)(h)
        h = layers.Dense(dim)(h)
        x = layers.Add()([x, h])
    x = layers.GlobalAveragePooling1D()(layers.LayerNormalization()(x))
    return keras.Model(inp, _head(x, n_classes, dropout), name="transformer")


BUILDERS = {"gru": build_gru, "lstm": build_lstm, "cnn": build_cnn, "transformer": build_transformer}


def build_model(name, seq_len, n_features, n_classes, dropout=0.3):
    if name not in BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(BUILDERS)}")
    return BUILDERS[name](seq_len, n_features, n_classes, dropout)
