import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim=64, num_heads=4, ff_dim=128):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.attn(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(input_shape=(10, 40)):  # 10 timesteps, 40 features (20 bids + 20 asks)
    inputs = tf.keras.Input(shape=input_shape)
    x = TransformerBlock(embed_dim=64, num_heads=4)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)  # Long, Short, Flat
    return tf.keras.Model(inputs=inputs, outputs=outputs)