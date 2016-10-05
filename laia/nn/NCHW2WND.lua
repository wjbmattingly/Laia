local NCHW2WND, parent = torch.class('laia.nn.NCHW2WND', 'nn.Module')

function NCHW2WND:__init()
  parent.__init(self)
end

function NCHW2WND:updateOutput(input)
  assert(input:nDimension() == 4)
  local N, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  self.output = input:permute(4, 1, 3, 2):contiguous()
  self.output = self.output:view(W, N, H * C)
  return self.output
end

function NCHW2WND:updateGradInput(input, gradOutput)
  assert(gradOutput:isSameSizeAs(self.output), 'gradOutput has incorrect size!')
  assert(input:nDimension() == 4)
  local N, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  self.gradInput = gradOutput:view(W, N, H, C):permute(2, 4, 3, 1):contiguous()
  return self.gradInput
end

return NCHW2WND
