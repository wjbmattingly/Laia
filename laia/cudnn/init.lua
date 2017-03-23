require 'cudnn'

local ffi = require 'ffi'
local errcheck = cudnn.errcheck

if not laia then laia = {} end
if not laia.cudnn then laia.cudnn = {} end

function laia.cudnn.createDescriptor(typename, size, stride)
  assert(cudnn.typemap[typename])
  local descriptor = ffi.new('struct cudnnTensorStruct*[1]')
  -- create descriptor
  errcheck('cudnnCreateTensorDescriptor', descriptor)
  -- set gc hook
  local function destroy(d)
    errcheck('cudnnDestroyTensorDescriptor', d[0]);
  end
  ffi.gc(descriptor, destroy)
  -- set descriptor
  errcheck('cudnnSetTensor4dDescriptorEx', descriptor[0],
	   cudnn.typemap[typename],
	   size[1], size[2], size[3], size[4],
	   stride[1], stride[2], stride[3], stride[4])
  return descriptor
end

require('laia.cudnn.PoolingHWNC')
require('laia.cudnn.SpatialMaxPoolingHWNC')
require('laia.cudnn.SpatialConvolutionHWNC')
