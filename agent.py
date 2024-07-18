import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, LayerNormalization, ReLU, Concatenate, Conv1D, Dropout
from tensorflow.python.keras.layers import Embedding

class ResourceEncoder(tf.keras.Model):
    def __init__(self, num_servers, num_processors, num_resources):
        super(ResourceEncoder, self).__init__()
        self.num_servers = num_servers
        self.num_processors = num_processors
        self.num_resources = num_resources
        self.conv1d_layers = [Conv1D(64, kernel_size=num_resources, strides=num_resources) for _ in range(num_processors)]
        self.fc = Dense(128, activation='relu')

    def call(self, ResourceState, training=False):
        # inputs: (batch_size, num_servers, num_processors, num_resources)
        x = ResourceState
        for i in range(self.num_processors):
            x = self.conv1d_layers[i](x, training=training)
        x = tf.reduce_mean(x, axis=2)  # (batch_size, num_servers, num_resources)
        x = tf.reshape(x, (-1, self.num_servers * self.num_resources))
        x = self.fc(x, training=training)
        return x

class MAGNA(tf.keras.layers.Layer):
    def __init__(self, num_features, num_classes, num_layers, num_heads, dropout_rate):
        super(MAGNA, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.embedding = Embedding(num_features, 64)

        self.attention_layers = [
            tf.keras.layers.Attention(num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ]

        self.ffn_layers = [
            tf.keras.Sequential([
                Dense(64, activation="relu", kernel_regularizer=l2(1e-5)),
                Dropout(dropout_rate),
                Dense(num_classes)
            ])
            for _ in range(num_layers)
        ]

    def call(self, inputs, adj_matrix, training=False):

        x = self.embedding(inputs)

        for i in range(self.num_layers):

            attention_weights = self.attention_layers[i]([x, x], mask=adj_matrix)
            x = tf.matmul(attention_weights, x, transpose_a=True)
            x = self.ffn_layers[i](x)
            x = LayerNormalization()(x)

        return x

class MAGNATaskEncoder(tf.keras.Model):
    def __init__(self, num_layers, num_heads, num_resources):
        super(MAGNATaskEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_resources = num_resources
        self.attention_layers = [MAGNA(num_heads=self.num_heads) for _ in range(num_layers)]
        self.feedforward_layers = [tf.keras.Sequential([
            Dense(512, activation='relu'),
            Dense(256)
        ]) for _ in range(num_layers)]
        self.aggregation_layer = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, Task, dependencies, training=False):
        # tasks: (batch_size, num_tasks, num_resources)
        # dependencies: (batch_size, num_tasks, num_tasks)
        x = Task
        for i in range(self.num_layers):
            x = self.attention_layers[i](x, x, x, training=training)
            x = self.feedforward_layers[i](x, training=training)

        attention_matrix = tf.nn.softmax(dependencies, axis=-1)
        aggregated_features = tf.matmul(attention_matrix, x)
        aggregated_features = self.aggregation_layer(aggregated_features)
        return aggregated_features

class KAN(tf.keras.Model):
    def __init__(self, num_layers, num_units):
        super(KAN, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.units = [tf.keras.layers.Dense(self.num_units) for _ in range(self.num_layers)]

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_layers):
            x = self.units[i](x, training=training)
        return x

class HAgent(tf.keras.Model):
    def __init__(self, num_layers, num_heads, num_resources, num_units):
        super(HAgent, self).__init__()
        self.task_encoder = MAGNATaskEncoder(num_layers, num_heads, num_resources)
        self.resource_encoder = ResourceEncoder(num_resources=num_resources)
        self.kan = tf.keras.Sequential([
            Input((num_resources + 1,)),
            Dense(num_units, activation='relu'),
            Dense(num_units)
        ])

    def call(self, tasks, resources, dependencies, training=False):
        task_encodings = self.task_encoder(tasks, dependencies, training=training)
        resource_encodings = self.resource_encoder(resources, training=training)
        task_resource_encodings = Concatenate()([task_encodings, resource_encodings])
        priorities = self.kan(task_resource_encodings, training=training)
        return priorities

class LAgent(tf.keras.Model):
    def __init__(self, num_resources, num_units):
        super(LAgent, self).__init__()
        self.q_network = tf.keras.Sequential([
            Input((num_resources + 1,)),
            Dense(num_units, activation='relu'),
            Dense(num_resources)
        ])
        self.target_q_network = tf.keras.Sequential([
            Input((num_resources + 1,)),
            Dense(num_units, activation='relu'),
            Dense(num_resources)
        ])

    def call(self, inputs, training=False):
        return self.q_network(inputs, training=training)

    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

class MDHRL(tf.keras.Model):
    def __init__(self, num_layers, num_heads, num_resources, num_units):
        super(MDHRL, self).__init__()
        self.h_agent = HAgent(num_layers, num_heads, num_resources, num_units)
        self.l_agent = LAgent(num_resources, num_units)

    def call(self, tasks, resources, dependencies, training=False):
        priorities = self.h_agent(tasks, resources, dependencies, training=training)
        actions = tf.argmax(priorities, axis=1)
        q_values = self.l_agent(tf.concat([resources, actions[:, None]], axis=1), training=training)
        return q_values, actions
