# Notes on Recurrent Neural Networks

# Frameworks

### Pytorch

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
