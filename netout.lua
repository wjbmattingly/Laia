#!/usr/bin/env th

require 'torch'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nn'

require 'laia.utilities'
require 'laia.RandomBatcher'; local Batcher = RandomBatcher

function opts_parse(arg)
  cmd = torch.CmdLine()

  cmd:text()
  cmd:text('Generate raw outputs of a DCNN-LSTM-CTC model.')
  cmd:text()

  cmd:text('Options:')
  cmd:option('-batch_size', 40, 'Batch size')
  cmd:option('-min_width', 0, 'Minimum image width for batches')
  cmd:option('-width_factor', true, 'Make width a factor of the max pooling reduction')
  cmd:option('-gpu', 0, 'Which gpu to use. -1 = use CPU')
  cmd:option('-seed', 0x12345, 'Random number generator seed to use')
  cmd:option('-htk', false, 'Output in HTK format')
  cmd:option('-ark', false, 'Output as Kaldi table of lattices in ASCII')
  cmd:option('-softmax', false, 'Whether to add softmax layer at end of network')
  cmd:option('-convout', false, 'Whether to provide output of convolutional layers')

  cmd:option('-loglkh', '', 'Compute log-likelihoods using provided priors')
  cmd:option('-alpha', 0.3, 'p(x|s) = P(s|x) / P(s)^LOGLKH_ALPHA')

  cmd:option('-maxseq', false, 'Whether to output the sequence of maximums')
  cmd:option('-forcealign', false, 'Do forced alignment using given ground truth')
  cmd:option('-priorcomp', false, 'Compute priors using given ground truth')
  cmd:option('-symbols_table', '', 'List of symbols (for forced alingment or prior computation)')
  cmd:option('-gt_file', '', 'Data ground truth (for forced alingment or prior computation)')

  cmd:option('-outpads', '', 'File to output image horizontal paddings')

  cmd:text()

  cmd:text('Arguments:')
  cmd:argument('model', 'Path to the neural network model file')
  cmd:argument('data', 'Path to the list of images')
  cmd:argument('output', 'Directory to write matrices or file for ark/maxseq/forcealign/priorcomp ("-" is stdout)')
  cmd:text()

  local opt = cmd:parse(arg or {})
  assert(opt.batch_size > 0, 'Batch size must be greater than 0')
  return opt
end

local opt = opts_parse(arg)

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

local model = torch.load(opt.model).model

-- Output file modes (ark, maxseq, forcealign and priorcomp)
local outfile = false
if opt.ark or opt.maxseq or opt.forcealign or opt.priorcomp then
  outfile = opt.output == '-' and io.stdout or io.open(opt.output, 'w')
  assert(outfile ~= nil, string.format('Unable open file for writing: %q', opt.output))
  opt.convout = false
  if not opt.ark then
    opt.loglkh = ''
  end
end

-- Forced aling and prior computation modes
local prior_count
if opt.forcealign or opt.priorcomp then
  assert(opt.gt_file ~= '', string.format('For %s the data ground truth is required', opt.forcealign and 'forcealign' or 'priorcomp' ))
  assert(opt.symbols_table ~= '', string.format('For %s the symbols table is required', opt.forcealign and 'forcealign' or 'priorcomp' ))
  opt.softmax = true
else
  opt.gt_file = nil
end

-- Log-likelihood computation mode
local loglkh = false
local logprior = {}
local zeroprior = {}
if opt.loglkh ~= '' then
  local f = io.open(opt.loglkh, 'r')
  assert(f ~= nil, string.format('Unable to read priors file: %q', opt.loglkh))
  local ln = 0
  while true do
    local line = f:read('*line')
    if line == nil then break end
    ln = ln+1
    line = line:match('^%s*(.-)%s*$'):gsub('.*%s','')
    local prior = tonumber(line)
    zeroprior[ln] = prior == 0 and true or false
    logprior[ln] = torch.log(prior)*opt.alpha
  end
  f:close()
  loglkh = true
  opt.softmax = true
end

-- Output image paddings
local outpads = false
if opt.outpads ~= '' then
  outpads = io.open(opt.outpads, 'w')
end

-- Output from convolutional layers
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

-- Add softmax layer
if opt.softmax then
  model:add(nn.SoftMax())
end

-- Prepare for CUDA or CPU
if opt.gpu >= 0 then
  cutorch.setDevice(opt.gpu + 1) -- +1 because lua is 1-indexed
  model = model:cuda()
  cudnn.convert(model, cudnn)
else
  -- CPU
  model:float()
end

model:evaluate()

-- Read input channels from the model
opt.channels = model:get(1):get(1).nInputPlane
-- Compute width factor from model
if opt.width_factor then
  opt.width_factor = 1
  local maxpool = model:findModules('cudnn.SpatialMaxPooling')
  for n=1,#maxpool do
    opt.width_factor = opt.width_factor * maxpool[n].kW
  end
else
  opt.width_factor = 0
end
local dv = Batcher(opt.data, opt); dv:epochReset()

local nSamples = torch.IntStorage(1);
local sampPeriod = torch.IntStorage(1); sampPeriod[1] = 100000; -- 10000000 = 1seg
local sampSize = torch.ShortStorage(1);
local parmKind = torch.ShortStorage(1); parmKind[1] = 9; -- PARMKIND=USER

local n = 0
for batch=1,dv:numSamples(),opt.batch_size do
  -- Prepare batch
  local batch_size = n+opt.batch_size > dv:numSamples() and dv:numSamples()-n or opt.batch_size
  local batch_img, batch_gt, _, batch_ids, batch_hpad = dv:next(opt.batch_size)
  if opt.gpu >= 0 then
    batch_img = batch_img:cuda()
  end

  if outpads then
    for i = 1, batch_size do
      outpads:write(batch_ids[i]..' '..batch_hpad[i][1]..' '..batch_hpad[i][2]..' '..batch_hpad[i][3]..'\n')
    end
    outpads:flush();
  end

  -- Forward through network
  local output = model:forward(batch_img)

  -- Output sequence of maximums
  if opt.maxseq then
    local maxval, maxidx = torch.max(output,2)
    maxval = maxval:squeeze();
    maxidx = maxidx:squeeze():csub(1);

    -- Loop through batch samples
    for i = 1, batch_size do
      outfile:write(batch_ids[i]..' -')
      for f=i,output:size(1),opt.batch_size do
        outfile:write(' '..maxidx[f]..':'..maxval[f])
      end
      outfile:write('\n')
    end

  -- Compute forced alignment and priors
  elseif opt.forcealign or opt.priorcomp then
    if opt.priorcomp and not prior_count then
      assert(dv:numSymbols()+1 == output:size(2), 'Number of symbols should match the network output')
      prior_count = torch.IntTensor(output:size(2)):zero()
    end

    local nframes = output:size(1) / opt.batch_size
    local sample = torch.FloatTensor(nframes,output:size(2))

    -- Loop through batch samples
    for i = 1, batch_size do

      -- Copy sample for speed
      local j = 1
      for f=i,output:size(1),opt.batch_size do
        sample[{j,{}}]:copy(output[{f,{}}])
        j = j+1
      end

      -- Do forced alignment of sample w.r.t. batch_gt[i]
      sample:log()
      local tbFA = forceAlignment(sample,batch_gt[i])

      if opt.forcealign then
        outfile:write(batch_ids[i])
        -- Print alignment
        for f=1,nframes do
          outfile:write(' '..(tbFA[f]-1))
        end
        outfile:write('\n')
      else
        -- Increment prior_count
        for _,v in pairs(tbFA) do
          prior_count[v] = prior_count[v] + 1
        end
      end
    end

  -- Ouput from convolutional layers
  elseif opt.convout then

    -- Loop through batch samples
    for i = 1, batch_size do
      -- Output in HTK format
      if opt.htk then
        local fd = torch.DiskFile( opt.output..'/'..batch_ids[i]..'.fea', 'w' ):binary():bigEndianEncoding()
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
        fd:close()
      -- Output in ASCII format
      else
        local fd = io.open( opt.output..'/'..batch_ids[i]..'.fea', 'wb' )
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
    local values = nframes*output:size(2)
    local sample = torch.FloatTensor(nframes,output:size(2))

    -- Softmax output
    --if opt.softmax2 then
    --  output:exp()
    --  output:cdiv( output:sum(2):repeatTensor(1,output:size(2)) );
    --end

    if loglkh then
      for k = 1, sample:size(2) do
        if zeroprior[k] then
          output[{{},k}]:fill(-743.747)
        else
          output[{{},k}]:log():csub(logprior[k])
        end
      end
    end

    -- Loop through batch samples
    for i = 1, batch_size do

      -- Copy sample for speed
      local j = 1
      for f=i,output:size(1),opt.batch_size do
        sample[{j,{}}]:copy(output[{f,{}}])
        j = j+1
      end

      -- Output in ARK format
      if opt.ark then
        if not loglkh then
          sample:log()
        end
        outfile:write( batch_ids[i]..'\n' )
        for j=1,sample:size(1) do
          for k = 1, sample:size(2) do
            outfile:write( string.format('%d %d %d 0,%g,\n',j-1,j,k,-sample[{j,k}]) )
          end
        end
        outfile:write( sample:size(1)..'\n\n' )

      -- Output in HTK format
      elseif opt.htk then
        local fd = torch.DiskFile( opt.output..'/'..batch_ids[i]..'.fea', 'w' ):binary():bigEndianEncoding()
        nSamples[1] = output:size(1)/opt.batch_size
        sampSize[1] = 4*output:size(2)
        fd:writeInt( nSamples[1] )
        fd:writeInt( sampPeriod[1] )
        fd:writeShort( sampSize[1] )
        fd:writeShort( parmKind[1] )
        local storage = sample:storage()
        for j=1,values do
          fd:writeFloat( storage[j] )
        end
        fd:close()
      -- Output in ASCII format
      else
        local fd = io.open( opt.output..'/'..batch_ids[i]..'.fea', 'wb' )
        for j=1,sample:size(1) do
          fd:write( string.format('%g',sample[{j,1}]) )
          for k = 2, sample:size(2) do
            fd:write( string.format(' %g',sample[{j,k}]) )
          end
          fd:write('\n')
        end
        fd:close()
      end
    end

  end

  n = n + batch_size
end

if outpads then
  outpads:close()
end

if opt.priorcomp then
  local prior_total = prior_count:sum()
  for n=1, prior_count:size(1) do
    outfile:write( string.format('%d\t%d\t%d\t%.10e\n',n-1,prior_count[n],prior_total,prior_count[n]/prior_total) )
  end
end
if outfile then
  outfile:close()
end
