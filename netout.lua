require 'torch'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'

require 'src.utilities'
require 'src.RandomBatcher'; local Batcher = RandomBatcher

function opts_parse(arg)
  cmd = torch.CmdLine()

  cmd:text()
  cmd:text('Generate raw outputs of a DCNN-LSTM-CTC model.')
  cmd:text()

  cmd:text('Options:')
  cmd:option('-batch_size', 40, 'Batch size')
  cmd:option('-min_width', 0, 'Minimum image width for batches')
  cmd:option('-gpu', 0, 'Which gpu to use. -1 = use CPU')
  cmd:option('-seed', 0x12345, 'Random number generator seed to use')
  --cmd:option('-symbols_table', '', 'Symbols table (original_symbols.txt)')
  cmd:option('-htk', false, 'Output in KTK format')
  -- @todo HTK format write time similar to ASCII, normal? can it be faster?
  cmd:option('-softmax', false, 'Whether to softmax output')
  cmd:option('-convout', false, 'Whether to provide output of convolutional layers')
  cmd:option('-outpads', false, 'Whether to output image horizontal paddings')
  cmd:text()

  cmd:text('Arguments:')
  cmd:argument('model', 'Path to the neural network model file')
  cmd:argument('data', 'Path to the list of images')
  cmd:argument('outdir', 'Directory to write character posterior matrices')
  cmd:text()

  local opt = cmd:parse(arg or {})
  assert(opt.batch_size > 0, 'Batch size must be greater than 0')
  return opt
end

local opt = opts_parse(arg)

math.randomseed(opt.seed)
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

local model = torch.load(opt.model).model

if opt.convout then
  local blstm = model:findModules('cudnn.BLSTM')
  assert(#blstm > 0, 'For convout expected to find a BLSTM module')
  local n = model:size()
  while model:get(n) ~= blstm[1] do
    model:remove()
    n = n-1
  end
  model:remove()
end

if opt.softmax then
  model:add(nn.SoftMax())
end

if opt.gpu >= 0 then
  cutorch.setDevice(opt.gpu + 1) -- +1 because lua is 1-indexed
  model = model:cuda()
  cudnn.convert(model, cudnn)
else
  -- CPU
  model:float()
end

-- Load symbols
--if opt.symbols_table then
--  local symbols_table = read_symbols_table(opt.symbols_table)
--end

model:evaluate()

-- Read input channels from the model
opt.channels = model:get(1):get(1).nInputPlane
-- Factor for batch widths
opt.width_factor = 8 -- @todo Add option for this and compute the value from the model
local dv = Batcher(opt.data, opt); dv:epochReset()

local nSamples = torch.IntStorage(1);
local sampPeriod = torch.IntStorage(1); sampPeriod[1] = 100000; -- 10000000 = 1seg
local sampSize = torch.ShortStorage(1);
local parmKind = torch.ShortStorage(1); parmKind[1] = 9; -- PARMKIND=USER

local n = 0
for batch=1,dv:numSamples(),opt.batch_size do
  -- Prepare batch
  local batch_size = n+opt.batch_size > dv:numSamples() and dv:numSamples()-n or opt.batch_size
  local batch_img, _, _, batch_ids, batch_hpad = dv:next(opt.batch_size)
  if opt.gpu >= 0 then
    batch_img = batch_img:cuda()
  end

  if opt.outpads then
    for i = 1, batch_size do
      io.write(batch_ids[i]..' '..batch_hpad[i][1]..' '..batch_hpad[i][2]..' '..batch_hpad[i][3]..'\n')
    end
  end

  -- Forward through network
  local output = model:forward(batch_img)

  -- Ouput from convolutional layers
  if opt.convout then

    for i = 1, batch_size do
      -- Output in HTK format
      if opt.htk then
        local fd = torch.DiskFile( opt.outdir..'/'..batch_ids[i]..'.fea', 'w' ):binary()
        fd:bigEndianEncoding()
        nSamples[1] = output:size(1)
        sampSize[1] = 4*output:size(3)
        fd:writeInt( nSamples[1] )
        fd:writeInt( sampPeriod[1] )
        fd:writeShort( sampSize[1] )
        fd:writeShort( parmKind[1] )
        for f=1, output:size(1) do
          for c = 1, output:size(3) do
            fd:writeFloat( output[f][i][c] )
          end
        end
      -- Output in ASCII format
      else
        local fd = io.open( opt.outdir..'/'..batch_ids[i]..'.fea', 'wb' )
        for f=1, output:size(1) do
          fd:write( string.format('%g',output[f][i][1]) )
          for c = 2, output:size(3) do
            fd:write( string.format(' %g',output[f][i][c]) )
          end
          fd:write('\n')
        end
        fd:close()
      end
    end

  -- Output from complete network
  else

    local nframes = output:size(1) / opt.batch_size
    -- Softmax output
    --if opt.softmax then
    --  output:exp()
    --  output:cdiv( output:sum(2):repeatTensor(1,output:size(2)) ); 
    --end
    local _, maxidx = torch.max(output,2)
    maxidx = maxidx:squeeze();

    local foff = 0
    for i = 1, batch_size do
      -- Output in HTK format
      if opt.htk then
        local fd = torch.DiskFile( opt.outdir..'/'..batch_ids[i]..'.fea', 'w' ):binary():bigEndianEncoding()
        nSamples[1] = output:size(1)/opt.batch_size
        sampSize[1] = 4*output:size(2)
        fd:writeInt( nSamples[1] )
        fd:writeInt( sampPeriod[1] )
        fd:writeShort( sampSize[1] )
        fd:writeShort( parmKind[1] )
        for f=i,output:size(1),opt.batch_size do
          for c = 1, output:size(2) do
            fd:writeFloat( output[f][c] )
          end
        end
      -- Output in ASCII format
      else
        local fd = io.open( opt.outdir..'/'..batch_ids[i]..'.fea', 'wb' )
        for f=i,output:size(1),opt.batch_size do
          --fd:write( string.format('%s :: ',symbols_table[maxidx[f]-1]) )
          fd:write( string.format('%g',output[f][1]) )
          for c = 2, output:size(2) do
            fd:write( string.format(' %g',output[f][c]) )
          end
          fd:write('\n')
        end
        fd:close()
      end
      foff = foff + nframes
    end

  end

  n = n + batch_size
end
