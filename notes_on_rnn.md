# Notes on Recurrent Neural Networks

# Frameworks

## Pytorch

#### Data format
- data.shape = (batch_size, sequence_length, input_size)
  - batch_size: size of batch.
  - sequence_length: the length of a sequence.
  - input_size: dimension of input

  
#### Key APIs
- torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
- torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)


#### Sequence Modes
- One-to-One
  - code snippet:
  ```
  x, hidden = self.lstm(x, hidden)
  # x.shape = (batch_size, sequence_length=1, input_size)
  output = x
  final_output= output[:, 0, :]
  ```
  
- Many-to-One
  - code snippet:
  ```
  x, hidden = self.lstm(x, hidden)
  # x.shape = (batch_size, sequence_length, input_size)
  output = x.view(batch_size, -1, input_size)
  final_output = x[:, -1, :]
  ```
  
- Many-to-Many
  - Similar to One-to-One
  - code snippet:
  ```
  x, hidden = self.lstm(x, hidden)
  output = x
  final_output = x[:, :, :]
  ```


## Keras
#### Data format
- The data format is same with PyTorch.
- data.shape = (batch_size, sequence_length, input_size)
  - batch_size: size of batch.
  - sequence_length: the length of a sequence.
  - input_size: dimension of input
  
  
#### Key APIs
- keras.layers.embeddings.Embedding(input_dim, output_dim, input_length=sequence_length)
- keras.layers.GRU(units=hidden_size, return_sequences=True)
- keras.layers.GRU(units=hidden_size, return_sequences=False)   # For many-to-one model
- keras.layers.RepeatVector(sequence_length)
- keras.layers.Bidirectional()
- keras.layers.TimeDistributed(keras.layers.Dense())


#### Sequence Modes
- One-to-One
  - code snippet:
  ```
  x_embed = keras.layers.embeddings.Embedding(input_dim=input_size, output_dim=embed_size, input_length=sequence_length)(x)
  x_rnn = keras.layes.GRU(units=hidden_size, return_sequences=True)(x_embed)
  # x.shape = (batch_size, 1, input_size)
  y = keras.layers.TimeDistributed(keras.layers.Dense(output_size, activation='softmax'))(x_rnn)
  ```
  
- Many-to-One
  - code snippet:
  ```
  x_embed = keras.layers.embeddings.Embedding(input_dim=input_size, output_dim=embed_size, input_length=sequence_length)(x)
  x_rnn = keras.layers.GRU(units=hidden_size, return_sequences=False)(x_embed)
  y = keras.layers.TimeDistributed(keras.layers.Dense(output_size, activation='softmax'))(x_rnn)
  ```
  
- Many-to-Many
  - Similar to One-to-One
  - code snippet:
  ```
  x_embed = keras.layers.embeddings.Embedding(input_dim=input_size, output_dim=embed_size, input_length=sequence_length)(x)
  x_rnn = keras.layers.GRU(units=hidden_size, return_sequences=True)(x_embed)
  y = keras.layers.TimeDistributed(keras.layers.Dense(output_size, activation='softmax'))(x_rnn)
  ```
  
- Encoder-Decoder
  - code snippet:
  ```
  x_embed = keras.layers.embeddings.Embedding(input_dim=input_size, output_dim=embed_size, input_length=sequence_length)(x)
  x_rnn_encoder = keras.layers.GRU(units=hidden_size, return_sequences=False)(x_bembed)
  
  x_rep = keras.layers.RepeatVector(sequence_length)(x_rnn_encoder)
  
  x_rnn_decoder = keras.layers.GRU(units=hidden_size, return_sequences=True)(x_rep)
  y = keras.layers.TimeDistributed(keras.layers.Dense(output_size, activation='softmax'))(x_rnn_decoder)
  ```
