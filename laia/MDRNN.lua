local MDRNN, parent = torch.class('laia.MDRNN', 'nn.Module')

function MDRNN:__init(inputSize, hiddenSize, rnn_type, dropout)
  parent.__init(self)
  rnn_type = rnn_type or 'lstm'
  assert(rnn_type == 'lstm' or rnn_type == 'gru')
  if rnn_type == 'lstm' then
    self.rnn_c = cudnn.BLSTM(inputSize, hiddenSize, 1, false, dropout)
    self.rnn_r = cudnn.BLSTM(inputSize, hiddenSize, 1, false, dropout)
  else
    self.rnn_c = cudnn.BGRU(inputSize, hiddenSize, 1, false, dropout)
    self.rnn_r = cudnn.BGRU(inputSize, hiddenSize, 1, false, dropout)
  end
  self.inputSize = inputSize
  self.hiddenSize = hiddenSize
end

function MDRNN:joinOutput(N, H, W, output_c, output_r)
  -- Wx(N*H)x(2*hiddenSize) -> WxNxHx(2*hiddenSize) -> Nx(2*hiddenSize)xHxW
  output_c = output_c:view(W, N, H, 2 * self.hiddenSize):permute(2, 4, 3, 1)
  -- Hx(N*W)x(2*hiddenSize) -> HxNxWx(2*hiddenSize) -> Nx(2*hiddenSize)xHxW
  output_r = output_r:view(H, N, W, 2 * self.hiddenSize):permute(2, 4, 1, 3)
  -- Nx(4*hiddenSize)xHxW
  return torch.cat(output_c, output_r, 2)
end

function MDRNN:joinGradInput(N, H, W, gradInput_c, gradInput_r)
  -- Wx(N*H)x(inputSize) -> WxNxHx(inputSize) -> Nx(inputSize)xHxW
  gradInput_c = gradInput_c:view(W, N, H, self.inputSize)
  gradInput_c = gradInput_c:permute(2, 4, 3, 1)
  -- Hx(N*W)x(inputSize) -> HxNxWx(inputSize) -> Nx(inputSize)xHxW
  gradInput_r = gradInput_r:view(H, N, W, self.inputSize)
  gradInput_r = gradInput_r:permute(2, 4, 1, 3)
  return gradInput_c + gradInput_r
end

function MDRNN:splitInput(input)
  local N, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  assert(C == self.inputSize, string.format(
	   'Wrong input size (expected = %d, actual = %d)', self.inputSize, C))
  -- Nx(inputSize)xHxW -> WxNxHx(inputSize) -> Wx(N*H)x(inputSize)
  local input_c =
    input:permute(4, 1, 3, 2):contiguous():view(W, N * H, self.inputSize)
  -- Nx(inputSize)xHxW -> HxNxWx(inputSize) -> Hx(N*W)x(inputSize)
  local input_r =
    input:permute(3, 1, 4, 2):contiguous():view(H, N * W, self.inputSize)
  return input_c, input_r
end

function MDRNN:splitGradOutput(output)
  local N, C, H, W =
    output:size(1), output:size(2), output:size(3), output:size(4)
  assert(self.hiddenSize * 4 == C, string.format(
	   'Wrong gradOutput size (expected = %d, actual = %d)',
	   4 * self.hiddenSize, C))
  local output_c = output:sub(1, N,
			      1, 2 * self.hiddenSize,
			      1, H, 1, W)
  -- Nx(2*hiddenSize)xHxW -> HxNxWx(2*hiddenSize) -> Hx(N*W)x(2*hiddenSize)
  output_c = output_c:permute(4, 1, 3, 2):contiguous()
  output_c = output_c:view(W, N * H, 2 * self.hiddenSize)

  local output_r = output:sub(1, N,
			      2 * self.hiddenSize + 1, 4 * self.hiddenSize,
			      1, H, 1, W)
  -- Nx(2*hiddenSize)xHxW -> WxNxHx(2*hiddenSize) -> Wx(N*H)x(2*hiddenSize)
  output_r = output_r:permute(3, 1, 4, 2):contiguous()
  output_r = output_r:view(H, N * W, 2 * self.hiddenSize)
  return output_c, output_r
end

function MDRNN:updateOutput(input)
  local N, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  assert(C == self.inputSize, string.format(
	   'Wrong input size (expected = %d, actual = %d)', self.inputSize, C))
  local input_c, input_r = self:splitInput(input)
  self.output = self:joinOutput(N, H, W,
				self.rnn_c:forward(input_c),
				self.rnn_r:forward(input_r))
  return self.output
end

function MDRNN:updateGradInput(input, gradOutput)
  assert(gradOutput:isSameSizeAs(self.output), 'gradOutput has incorrect size!')
  local N, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  assert(C == self.inputSize, string.format(
	   'Wrong input size (expected = %d, actual = %d)', self.inputSize, C))
  local input_c, input_r = self:splitInput(input)
  local gradOutput_c, gradOutput_r = self:splitGradOutput(gradOutput)
  self.gradInput =
    self:joinGradInput(N, H, W,
		       self.rnn_c:updateGradInput(input_c, gradOutput_c),
		       self.rnn_r:updateGradInput(input_r, gradOutput_r))
  return self.gradInput
end

function MDRNN:accGradParameters(input, gradOutput, scale)
  local input_c, input_r = self:splitInput(input)
  local gradOutput_c, gradOutput_r = self:splitGradOutput(gradOutput)
  self.rnn_c:accGradParameters(input_c, gradOutput_c, scale)
  self.rnn_r:accGradParameters(input_r, gradOutput_r, scale)
end

function MDRNN:zeroGradParameters()
  self.rnn_c:zeroGradParameters()
  self.rnn_r:zeroGradParameters()
end

function MDRNN:updateParameters(lr)
  self.rnn_c:updateParameters(lr)
  self.rnn_r:updateParameters(lr)
end

function MDRNN:training()
  self.rnn_c:training()
  self.rnn_r:training()
end

function MDRNN:evaluate()
  self.rnn_c:evaluate()
  self.rnn_r:evaluate()
end

function MDRNN:reset(stdv)
  self.rnn_c:reset(stdv)
  self.rnn_r:reset(stdv)
end

function MDRNN:parameters()
  local w = {self.rnn_c.weight, self.rnn_r.weight}
  local gw = {self.rnn_c.gradWeight, self.rnn_r.gradWeight}
  return w, gw
end

function MDRNN:clearState()
  self.output = nil
  self.gradInput = nil
  self.rnn_c:clearState()
  self.rnn_r:clearState()
  return self
end

return MDRNN
