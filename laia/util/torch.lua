torch = require 'torch'
laia = laia or {}

local ffi = wrequire 'ffi'
local ssllib = nil
if ffi ~= nil then
  ssllib = ffi.load('ssl')
  assert(ssllib ~= nil)
  ffi.cdef[[
typedef struct MD5state_st {
  unsigned long A,B,C,D;
  unsigned long Nl,Nh;
  unsigned long data[16];
  unsigned int num;
} MD5_CTX;
int MD5_Init(MD5_CTX *c);
int MD5_Update(MD5_CTX *c, const void *data, size_t len);
int MD5_Final(unsigned char *md, MD5_CTX *c);
unsigned char *MD5(const unsigned char *d, size_t n, unsigned char *md);
void MD5_Transform(MD5_CTX *c, const unsigned char *b);
]]
end

function laia.tensor_md5(x)
  assert(ffi ~= nil, 'FFI module was not found, you cannot use TensorMD5!')
  local typemap = {
    ['torch.CudaByteTensor'] = 'torch.ByteTensor',
    ['torch.CudaCharTensor'] = 'torch.CharTensor',
    ['torch.CudaShortTensor'] = 'torch.ShortTensor',
    ['torch.CudaIntTensor'] = 'torch.IntTensor',
    ['torch.CudaLongTensor'] = 'torch.LongTensor',
    ['torch.CudaTensor'] = 'torch.FloatTensor',
    ['torch.CudaDoubleTensor'] = 'torch.DoubleTensor',
    -- CPU tensors do not change type
    ['torch.ByteTensor'] = 'torch.ByteTensor',
    ['torch.CharTensor'] = 'torch.CharTensor',
    ['torch.ShortTensor'] = 'torch.ShortTensor',
    ['torch.IntTensor'] = 'torch.IntTensor',
    ['torch.LongTensor'] = 'torch.LongTensor',
    ['torch.FloatTensor'] = 'torch.FloatTensor',
    ['torch.DoubleTensor'] = 'torch.DoubleTensor'
  }
  local t = typemap[torch.type(x)]
  assert(t ~= nil,
	 ('Tensor type %q cannot be copied to the CPU'):format(torch.type(x)))
  -- Copy from GPU to CPU and make data contiguous (if necessary)
  x = x:type(t):contiguous()
  local memsize = x:nElement() * x:storage().elementSize()
  -- Compute MD5
  local ctx = ffi.new('MD5_CTX')
  local hash = ffi.new('unsigned char[16]')
  ssllib.MD5_Init(ctx)
  ssllib.MD5_Update(ctx, torch.data(x), memsize)
  ssllib.MD5_Final(hash, ctx)
  -- Return MD5 as an hex string
  local s = ffi.string(hash, 16)
  s = table.pack(s:byte(1, 16))
  local o = ''
  for i=1,#s do o = o .. ('%02x'):format(s[i]) end
  return o
end

-- Return a flat view of a nn.Module parameters and gradients.
-- This assumes that m:getParameters() was called before to flatten
-- the parameters and gradients of the model.
function laia.getFlatParameters(m)
  assert(m ~= nil and torch.isTypeOf(m, 'nn.Module'),
	 ('Expected a nn.Module class (type = %q)'):format(torch.type(m)))
  local Tensor = torch.factory(m:type())
  p, g = m:parameters()
  if not p then laia.log.warn('p is nil!!!!') end
  if not p or #p == 0 then return Tensor() end
  local sp, gp = p[1]:storage(), g[1]:storage()
  local n  = sp:size()
  return Tensor(sp, 1, n, 1), Tensor(gp, 1, n, 1)
end

-- Return the factor that divides the width of the image.
-- For now, it only works for pooling layers, and assuming that there is no
-- stride in these layers.
-- TODO(mauvilsa): Make this more generic for all types of layers.
local __getWidthFactorFromLayer = {
  ['cudnn.AveragePooling'] = function(m) return m.kW end,
  ['cudnn.SpatialMaxPooling'] = function(m) return m.kW end,
  ['nn.AveragePooling'] = function(m) return m.kW end,
  ['nn.SpatialMaxPooling'] = function(m) return m.kW end,
  ['nn.SpatialSubSampling'] = function(m) return m.kW end,
  ['nn.SpatialDilatedMaxPooling'] = function(m) return m.kW end,
  ['nn.SpatialLPPooling'] = function(m) return m.kW end
}
function laia.getWidthFactor(m)
  local factor = 1
  for t, func in pairs(__getWidthFactorFromLayer) do
    for _, layer in ipairs(m:findModules(t)) do
      factor = factor * func(layer)
    end
  end
  return factor
end
