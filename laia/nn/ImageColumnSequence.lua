local ImageColumnSequence, parent = torch.class('laia.nn.ImageColumnSequence',
						'nn.Module')

function ImageColumnSequence:__init()
  parent.__init(self)
end

function ImageColumnSequence:updateOutput(input)
  assert(input:nDimension() == 4, 'input must have 4 dimensions: N x C x H x W')
  local N, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  self.output = input:permute(4, 1, 3, 2):contiguous()
  self.output = self.output:view(W, N, H * C)
  return self.output
end

function ImageColumnSequence:updateGradInput(input, gradOutput)
  assert(input:nDimension() == 4, 'input must have 4 dimensions: N x C x H x W')
  assert(gradOutput:isSameSizeAs(self.output), 'gradOutput has incorrect size!')
  local N, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  self.gradInput = gradOutput:view(W, N, H, C):permute(2, 4, 3, 1):contiguous()
  return self.gradInput
end

return ImageColumnSequence
