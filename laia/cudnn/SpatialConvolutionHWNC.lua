local SpatialConvolution, parent =
    torch.class('laia.cudnn.SpatialConvolutionHWNC', 'nn.SpatialConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = cudnn.errcheck
local checkedCall = find.checkedCall

local sharedGradOutput = {}
local function getSharedGradOutput(gradOutput, force_copy, device, stream)
  force_copy = force_copy or false
  device = device or cutorch.getDevice()
  stream = stream or cutorch.getStream() -- starts from 0
  if not sharedGradOutput[device] then sharedGradOutput[device] = {} end
  local buf = sharedGradOutput[device][stream]
  if not buf or buf:type() ~= gradOutput:type() then
    buf = gradOutput:clone()
    sharedGradOutput[device][stream] = buf
  elseif not buf:isSameSizeAs(gradOutput) or force_copy then
    buf:resizeAs(gradOutput):copy(gradOutput)
  end
  return buf
end

function SpatialConvolution:__init(nInputPlane, nOutputPlane,
				   kW, kH, dW, dH, padW, padH)
  local delayedReset = self.reset
  self.reset = function() end
  parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
  self.reset = delayedReset
  self.padW = padW or 0
  self.padH = padH or 0
  self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
  self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
  self:reset()
  -- should nil for serialization, the reset will still work
  self.reset = nil
end

-- if you change the configuration of the module manually, call this
function SpatialConvolution:resetWeightDescriptors(desc)
  assert(cudnn.typemap[torch.typename(self.weight)], 'Only Cuda supported duh!')
  assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias,
	 'Only Cuda supported duh!')

  -- create descriptor for bias
  if self.bias then
    self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
  end

  self.weightDesc = cudnn.setFilterDescriptor(
    { dataType = cudnn.typemap[torch.typename(self.weight)],
      filterDimA = desc or
	{self.nOutputPlane,
	 self.nInputPlane,
	 self.kH, self.kW}
    }
  )

  return self
end

function SpatialConvolution:fastest(mode)
  if mode == nil then mode = true end
  if not self.fastest_mode or self.fastest_mode ~= mode then
    self.fastest_mode = mode
    self.iDesc = nil
  end
  return self
end

function SpatialConvolution:setMode(fmode, bdmode, bwmode)
  if fmode ~= nil then
    self.fmode = fmode
  end
  if bdmode ~= nil then
    self.bdmode = bdmode
  end
  if bwmode ~= nil then
    self.bwmode = bwmode
  end
  self.iDesc = nil
  return self
end

function SpatialConvolution:resetMode()
  self.fmode = nil
  self.bdmode = nil
  self.bwmode = nil
  return self
end

function SpatialConvolution:noBias()
  self.bias = nil
  self.gradBias = nil
  return self
end


function SpatialConvolution:checkInputChanged(input)
  assert(input:isContiguous(),
	 "input to " .. torch.type(self) .. " needs to be contiguous, but is non-contiguous")
  if not self.iSize or self.iSize:size() ~= input:dim() then
    self.iSize = torch.LongStorage(input:dim()):fill(0)
  end
  if not self.weightDesc then self:resetWeightDescriptors() end
  if not self.weightDesc then error "Weights not assigned!" end

  if not self.iDesc or not self.oDesc or input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
  or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
    self.iSize = input:size()
    assert(self.nInputPlane == input:size(4),
	   'input has to contain: '
	     .. self.nInputPlane
	     .. ' feature maps, but received input of size: '
	     .. input:size(1) .. ' x ' .. input:size(2) .. ' x ' .. input:size(3) .. ' x ' .. input:size(4))
    return true
  end
  return false
end

function SpatialConvolution:createIODescriptors(input)
  local batch = true
  if input:dim() == 3 then
    input = input:view(input:size(1), input:size(2), 1, input:size(3))
    batch = false
  end
  if SpatialConvolution.checkInputChanged(self, input) then
    -- create input descriptor
    self.iDesc = laia.cudnn.createDescriptor(
      input:type(),
      {
	input:size(3),    -- size N
	input:size(4),    -- size C
	input:size(1),    -- size H
	input:size(2),	  -- size W
      },
      {
	input:stride(3),  -- stride N
	input:stride(4),  -- stride C
	input:stride(1),  -- stride H
	input:stride(2),  -- stride W
      }
    )
    -- create conv descriptor
    self.padH, self.padW = self.padH or 0, self.padW or 0
    -- those needed to calculate hash
    self.pad = {self.padH, self.padW}
    self.stride = {self.dH, self.dW}

    self.convDescData = { padA = self.pad,
			  filterStrideA = self.stride,
			  upscaleA = {1,1},
			  dataType = cudnn.configmap(torch.type(self.weight))
    }
    self.convDesc = cudnn.setConvolutionDescriptor(self.convDescData)

    -- Get output shape (NxCxHxW layout)
    local oSize = torch.IntTensor(4)
    errcheck('cudnnGetConvolutionNdForwardOutputDim',
	     self.convDesc[0], self.iDesc[0],
	     self.weightDesc[0], 4, oSize:data())
    -- Resize output: HxWxNxC layout
    self.output:resize(oSize[3], oSize[4], oSize[1], oSize[2])
    self.oSize = self.output:size()

    -- create descriptor for output (HxWxNxC)
    self.oDesc = laia.cudnn.createDescriptor(
      self.output:type(),
      {
	self.output:size(3),
	self.output:size(4),
	self.output:size(1),
	self.output:size(2),
      },
      {
	self.output:stride(3),
	self.output:stride(4),
	self.output:stride(1),
	self.output:stride(2),
      }
    )
    -- create descriptor for outputGrad (NxCxHxW)
    self.goDesc = laia.cudnn.createDescriptor(
      self.output:type(),
      {
	self.output:size(3),  -- size N
	self.output:size(4),  -- size C
	self.output:size(1),  -- size H
	self.output:size(2),  -- size W
      },
      {
	self.output:size(4) * self.output:size(1) * self.output:size(2),
	self.output:size(1) * self.output:size(2),
	self.output:size(2),  -- stride C
	1,                    -- stride W
      }
    )

    find:prepare(self, input, self.output)

    if not batch then
      self.output = self.output:view(self.output:size(1),
				     self.output:size(2),
				     self.output:size(4))
    end
  end
  return self
end

function SpatialConvolution:updateOutput(input)
  assert(input:isContiguous())
  self:createIODescriptors(input)
  local finder = find.get()
  local fwdAlgo = finder:forwardAlgorithm(
    self, { self.iDesc[0], self.input_slice, self.weightDesc[0], self.weight,
	    self.convDesc[0], self.oDesc[0], self.output_slice})
  local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()

  checkedCall(self,'cudnnConvolutionForward', cudnn.getHandle(),
	      cudnn.scalar(input, 1),
	      self.iDesc[0], input:data(),
	      self.weightDesc[0], self.weight:data(),
	      self.convDesc[0], fwdAlgo,
	      extraBuffer, extraBufferSize,
	      cudnn.scalar(input, 0),
	      self.oDesc[0], self.output:data());

  -- add bias
  if self.bias then
    errcheck('cudnnAddTensor', cudnn.getHandle(),
	     cudnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
	     cudnn.scalar(input, 1), self.oDesc[0], self.output:data())
  end

  return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
  assert(input:isContiguous())
  assert(gradOutput:isContiguous())
  if not self.gradInput then return end
  self.gradInput:resizeAs(input)
  assert(gradOutput:dim() == input:dim())

  -- Change the view of the gradOutput to have 4 dimensions.
  if gradOutput:dim() == 2 then
    gradOutput = gradOutput:view(gradOutput:size(1), gradOutput:size(2), 1, 1)
  elseif gradOutput:dim() == 3 then
    gradOutput = gradOutput:view(gradOutput:size(1), gradOutput:size(2),
				 1, gradOutput:size(3))
  end
  -- Permute gradOutput to NxCxHxW since it is the only format accepted by
  -- cudnn for backpropagation. A (contiguous) copy of the permuted
  -- gradOutput is stored into a shared space in order to save memory across
  -- different layers.
  gradOutput = getSharedGradOutput(gradOutput:permute(3, 4, 1, 2), true)

  self:createIODescriptors(input)
  local finder = find.get()
  local bwdDataAlgo = finder:backwardDataAlgorithm(
    self, { self.weightDesc[0], self.weight,
	    self.goDesc[0], gradOutput,
	    self.convDesc[0], self.iDesc[0], self.input_slice })
  local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()

  checkedCall(self,'cudnnConvolutionBackwardData', cudnn.getHandle(),
	      cudnn.scalar(input, 1),
	      self.weightDesc[0], self.weight:data(),
	      self.goDesc[0], gradOutput:data(),
	      self.convDesc[0],
	      bwdDataAlgo,
	      extraBuffer, extraBufferSize,
	      cudnn.scalar(input, 0),
	      self.iDesc[0], self.gradInput:data())

  return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
  assert(input:isContiguous())
  assert(gradOutput:isContiguous())
  self.scaleT = self.scaleT or self.weight.new(1)
  -- this line forces this member to always be on CPU (needed for cudnn)
  self.scaleT = torch.type(self.weight) == 'torch.CudaDoubleTensor'
    and self.scaleT:double() or self.scaleT:float()
  scale = scale or 1.0
  self.scaleT[1] = scale

  -- Get the shared copy of the gradOutput.
  -- Notice that we are not forcing the copy to the shared memory, since
  -- this copy should have been done by the updateGradInput() method, which
  -- is always called before accGradParameters()
  getSharedGradOutput(gradOutput:permute(3, 4, 1, 2), false)

  self:createIODescriptors(input)
  local finder = find.get()
  local bwdFilterAlgo = finder:backwardFilterAlgorithm(
    self, {self.iDesc[0], self.input_slice, self.goDesc[0], gradOutput,
	   self.convDesc[0], self.weightDesc[0], self.weight})

  -- gradBias
  if self.bias then
    errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
	     self.scaleT:data(),
	     self.goDesc[0], gradOutput:data(),
	     cudnn.scalar(input, 1),
	     self.biasDesc[0], self.gradBias:data())
  end

  -- gradWeight
  local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
  checkedCall(self,'cudnnConvolutionBackwardFilter', cudnn.getHandle(),
	      self.scaleT:data(),
	      self.iDesc[0], input:data(),
	      self.goDesc[0], gradOutput:data(),
	      self.convDesc[0],
	      bwdFilterAlgo,
	      extraBuffer, extraBufferSize,
	      cudnn.scalar(input, 1),
	      self.weightDesc[0], self.gradWeight:data())
end

function SpatialConvolution:clearDesc()
  self.weightDesc = nil
  self.biasDesc = nil
  self.convDesc = nil
  self.iDesc = nil
  self.oDesc = nil
  self.goDesc = nil
  self.oSize = nil
  self.scaleT = nil
  return self
end

function SpatialConvolution:write(f)
  self:clearDesc()
  local var = {}
  for k,v in pairs(self) do
    var[k] = v
  end
  f:writeObject(var)
end

function SpatialConvolution:clearState()
  self:clearDesc()
  nn.utils.clear(self, 'input_slice', 'output_slice')
  return nn.Module.clearState(self)
end
