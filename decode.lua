require 'torch'
require 'cudnn'

require 'src.utilities'
require 'src.RandomBatcher'; local Batcher = RandomBatcher

-- local str2bool_table = {
--    ['true'] = true, ['false'] = false,
--    ['t'] = true, ['f'] = false,
--    ['True'] = true, ['False'] = false,
--    ['1'] = true, ['0'] = false,
--    ['TRUE'] = true, ['FALSE'] = false
-- }

local opts = require 'src.DecodeOptions'
local opt = opts.parse(arg)

math.randomseed(opt.seed)
torch.manualSeed(opt.seed)
if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
end

local model = torch.load(opt.model).model

if opt.gpu >= 0 then
  cutorch.setDevice(opt.gpu + 1) -- +1 because lua is 1-indexed
  model = model:cuda()
  cudnn.convert(model, cudnn)
else
  -- CPU
  model:float()
end

-- Load symbols
if opt.symbols_table then
  local symbols_table = read_symbols_table(opt.symbols_table)
end

model:evaluate()

-- Read input channels from the model
opt.channels = model:get(1):get(1).nInputPlane
-- Factor for batch widths
opt.width_factor = 8 -- @todo Add option for this and compute the value from the model
local dv = Batcher(opt.data, opt); dv:epochReset()
local n = 0
for batch=1,dv:numSamples(),opt.batch_size do
  -- Prepare batch
  local batch_img, _, _, batch_ids = dv:next(opt.batch_size)
  if opt.gpu >= 0 then
    batch_img = batch_img:cuda()
  end
  -- Forward through network
  local output = model:forward(batch_img)
  local batch_decode = framewise_decode(opt.batch_size, output)
  for i=1,opt.batch_size do
    n = n + 1
    -- Batch can contain more images
    if n > dv:numSamples() then
      break
    end
    io.write(string.format('%s', batch_ids[i]))
    for t=1, #batch_decode[i] do
      if opt.symbols_table ~= '' then
        -- Print symbols
        io.write(string.format(' %s', symbols_table[batch_decode[i][t]]))
      else
        -- Print id's
        io.write(string.format(' %d', batch_decode[i][t]))
      end
    end
    io.write('\n')
  end
end
