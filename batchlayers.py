import tensorflow as tf

# Function to convert batch input data to use scan ops of tensorflow.
def process_batch_input_for_RNN(batch_input):
    """
    Process tensor of size [5,3,2] to [3,5,2]
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    X = tf.transpose(batch_input_)

    return X

class LSTMLayerBatch(object):

    def __init__(self, input_size, hidden_layer_size, input_placeholder):

        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size

        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(tf.zeros(
            [self.input_size, self.hidden_layer_size]))
        self.Ui = tf.Variable(tf.zeros(
            [self.hidden_layer_size, self.hidden_layer_size]))
        self.bi = tf.Variable(tf.zeros([self.hidden_layer_size]))


        self.Wf = tf.Variable(tf.zeros(
            [self.input_size, self.hidden_layer_size]))
        self.Uf = tf.Variable(tf.zeros(
            [self.hidden_layer_size, self.hidden_layer_size]))
        self.bf = tf.Variable(tf.zeros([self.hidden_layer_size]))


        self.Wog = tf.Variable(tf.zeros(
            [self.input_size, self.hidden_layer_size]))
        self.Uog = tf.Variable(tf.zeros(
            [self.hidden_layer_size, self.hidden_layer_size]))
        self.bog = tf.Variable(tf.zeros([self.hidden_layer_size]))


        self.Wc = tf.Variable(tf.zeros(
            [self.input_size, self.hidden_layer_size]))
        self.Uc = tf.Variable(tf.zeros(
            [self.hidden_layer_size, self.hidden_layer_size]))
        self.bc = tf.Variable(tf.zeros([self.hidden_layer_size]))

        # Placeholder for input vector with shape[batch, seq, embeddings]
        self._inputs = input_placeholder

        # Processing inputs to work with scan function
        self.processed_input = process_batch_input_for_RNN(self._inputs)

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden= tf.matmul(
            self.initial_hidden, tf.zeros([input_size, hidden_layer_size]))


        self.initial_hidden=tf.pack([self.initial_hidden,self.initial_hidden])

    # Function for LSTM cell.
    def forward_step(self, previous_hidden_memory_tuple, x):
        """
        This function takes previous hidden state and memory tuple with input and
        outputs current hidden state.
        """

        previous_hidden_state,c_prev=tf.unpack(previous_hidden_memory_tuple)

        #Input Gate
        i= tf.sigmoid(
            tf.matmul(x,self.Wi)+tf.matmul(previous_hidden_state,self.Ui) + self.bi
        )

        #Forget Gate
        f= tf.sigmoid(
            tf.matmul(x,self.Wf)+tf.matmul(previous_hidden_state,self.Uf) + self.bf
        )

        #Output Gate
        o= tf.sigmoid(
            tf.matmul(x,self.Wog)+tf.matmul(previous_hidden_state,self.Uog) + self.bog
        )

        #New Memory Cell
        c_= tf.nn.tanh(
            tf.matmul(x,self.Wc)+tf.matmul(previous_hidden_state,self.Uc) + self.bc
        )

        #Final Memory cell
        c= f*c_prev + i*c_

        #Current Hidden state
        current_hidden_state = o*tf.nn.tanh(c)


        return tf.pack([current_hidden_state,c])

    # Function for getting all hidden state.
    def get_outputs(self):
        """
        Iterates through time/ sequence to get all hidden state
        """

        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self.forward_step,
                                    self.processed_input,
                                    initializer=self.initial_hidden,
                                    name='states')
        all_hidden_states=all_hidden_states[:,0,:,:]

        return all_hidden_states

class MeanLayerBatch(object):

    def __init__(self, input_layer):
        self.inputs = input_layer.get_outputs()

    def get_outputs(self):
        return tf.reduce_mean(self.inputs, reduction_indices=0)

class RegLayerBatch(object):

    def __init__(self, input_size, output_size, input_layer):
        self.inputs = input_layer.get_outputs()
        self.input_size = input_size
        self.output_size = output_size
        self.Wo = tf.Variable(tf.truncated_normal([self.input_size, self.output_size], mean=0, stddev=.01))
        self.bo = tf.Variable(tf.truncated_normal([self.output_size], mean=0, stddev=.01))

    def get_outputs(self):
        output = tf.matmul(self.inputs, self.Wo) + self.bo
        return output
