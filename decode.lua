require 'torch'
require 'cudnn'
require 'src.Model'

require 'src.utilities'
require 'src.WidthBatcher'

local str2bool_table = {
   ['true'] = true, ['false'] = false,
   ['t'] = true, ['f'] = false,
   ['True'] = true, ['False'] = false,
   ['1'] = true, ['0'] = false,
   ['TRUE'] = true, ['FALSE'] = false
}

local argparse = require 'argparse'
local parser = argparse('decode.lua', '')
parser:argument('model', 'Path to the neural network model file')
parser:argument('data', 'Path to the dataset HDF5 file')
parser:option('-b --batch_size', 'Batch size', 16):convert(tonumber)
parser:option('--use_gpu',
              'If true, perform computations on a GPU card supporting CUDA',
              'true'):convert(str2bool_table)
parser:option('--use_cudnn',
              'If true, use NVIDIA cuDNN toolkit',
              'true'):convert(str2bool_table)
local args = parser:parse()

if args.use_gpu then
  require 'cutorch'
  require 'cunn'
  if args.use_cudnn then 
    require 'cudnn' 
  end
end

local model = torch.load(args.model)

if args.use_gpu then
  model = model:cuda()
  if use_cudnn then 
    cudnn.convert(model, cudnn) 
  end
end
model:evaluate()

local dv = WidthBatcher(args.data, true)
local n = 0
for batch=1,dv:numSamples(),args.batch_size do
  -- Prepare batch
  local batch_img, _, _, batch_ids = dv:next(args.batch_size)
  if args.use_gpu then 
    batch_img = batch_img:cuda() 
  end
  -- Forward through network
  local output = model:forward(batch_img)
  local batch_decode = framewise_decode(args.batch_size, output)
  for i=1, args.batch_size do
    n = n + 1
    if n > dv:numSamples() then 
      break 
    end
    io.write(string.format('%s  ', batch_ids[i]))
    for t=1, #batch_decode[i] do
      io.write(string.format(' %d', batch_decode[i][t]))
    end
    io.write('\n')
  end
end
