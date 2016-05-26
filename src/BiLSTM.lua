local BiLSTM, parent = torch.class('cudnn.BiLSTM', 'cudnn.RNN')

function BiLSTM:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst)
    self.bidirectional = 'CUDNN_BIDIRECTIONAL'
    self.mode = 'CUDNN_LSTM'
    self.numDirections = 2
    self.dropout = dropout or 0
    self:reset()
end