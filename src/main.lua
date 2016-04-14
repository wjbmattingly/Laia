require 'argparse'
require 'torch'
require 'warp_ctc'
require 'optim'
require 'xlua'

require 'WidthBatcher'
require 'CurriculumBatcher'
require 'RandomBatcher'
require 'Utils'
require 'Model'

local seed = 1234
local use_gpu = true
local use_cudnn = true
local BATCH_SIZE = 50
local SAMPLE_HEIGHT = 64
local NUM_CHARS = 78
local grad_clip = 3

local rmsprop_opts = {
   learningRate = 0.001,
   alpha = 0.95
}
local learning_rate_decay = 0.97
local learning_rate_decay_after = 10
local curriculum_lambda_start = 3
local curriculum_lambda_iters = 10
local l1_decay = 0.001

if use_gpu then
  require 'cutorch'
  require 'cunn'
  if use_cudnn then 
    require 'cudnn' 
  end
end

torch.manualSeed(seed)
math.randomseed(seed)

id = os.date("%d.%m.%H.%M")

local model = createModel(SAMPLE_HEIGHT, NUM_CHARS)
if use_gpu then
  model = model:cuda()
  if use_cudnn then 
    cudnn.convert(model, cudnn) 
  end
end

parameters, gradParameters = model:getParameters()

function printHistogram(x, nbins, bmin, bmax)
  local hist, bins = torch.loghistc(x, nbins, bmin, bmax)
  local n = x:storage():size()
  io.write(string.format('(-inf, %g] -> %.2g%%\n',
            bins[1], 100 * hist[1] / n))
  for i=2,#hist do
    io.write(string.format('(%g, %g] -> %.2g%%\n',
              bins[i-1], bins[i], 100 * hist[i] / n))
  end
end


-- local dt = WidthBatcher('../data/iam/train.h5', true)
-- local dt = RandomBatcher('../data/iam/train.h5', true)
-- dt:shuffle()
-- local dt = CurriculumBatcher('../data/iam/train.h5', true)
-- dt:sample(3, 5)

local dt = CurriculumBatcher('../data/iam/train.h5', true)
local dv = RandomBatcher('../data/iam/valid.h5', true)

for epoch=1,1000 do
  -- Apply learning rate decay
  if epoch > learning_rate_decay_after then
    rmsprop_opts.learningRate =
    rmsprop_opts.learningRate * learning_rate_decay
  end
  -- Curriculum learning
  if epoch <= curriculum_lambda_iters then
    dt:sample(curriculum_lambda_start *
              (1.0 - (epoch - 1) / curriculum_lambda_iters), 5)
  else
    dt:sample(0, 5)
  end

  model:training()

  local train_loss_epoch = 0.0
  for batch=1,dt:numSamples(),BATCH_SIZE do
    local batch_img, batch_gt, batch_sizes = dt:next(BATCH_SIZE)
    
    if use_gpu then
     batch_img = batch_img:cuda()
    end

    local feval = function(x)
      assert (x == parameters)
      collectgarbage()
      gradParameters:zero()

      local output = model:forward(batch_img)
      local sizes = {}
      local seq_len = output:size()[1] / BATCH_SIZE
      for i=1,BATCH_SIZE do table.insert(sizes, seq_len) end

      local grad_output = output:clone():zero()
      local loss = 0
      
      -- Compute loss function and gradients respect the output
      if use_gpu then
        loss = table.reduce(gpu_ctc(output, grad_output, batch_gt, sizes),
                              operator.add, 0)
      else
        output = output:float()
        grad_output = grad_output:float()
        loss = table.reduce(cpu_ctc(output, grad_output, batch_gt, sizes),
                              operator.add, 0)
      end
      
      -- Make loss function (and output gradients) independent of batch size
      -- and sequence length.

      loss = loss / (BATCH_SIZE * seq_len)
      grad_output:div(BATCH_SIZE * seq_len)
      
      -- Compute gradients of the loss function respect the parameters
      model:backward(batch_img, grad_output)
      
      -- L1 Normalization
      -- gradParams:add(torch.sign(parameters):mul(l1_decay))
     
      -- Clip gradients
      if grad_clip > 0 then
        gradParameters:clamp(-grad_clip, grad_clip)
      end
      
      train_loss_epoch = train_loss_epoch + BATCH_SIZE * loss
      return loss, gradParameters
    end

    optim.rmsprop(feval, parameters, rmsprop_opts)
    xlua.progress(batch + BATCH_SIZE - 1, dt:numSamples())
  end
  
  local lastGradParameters = gradParameters:clone()

  -- VALIDATION
  model:evaluate()
  dv:shuffle()
  local best_valid_loss_epoch = 10.0 -- very "low" value
  local valid_loss_epoch = 0.0
  for batch=1,dv:numSamples(),BATCH_SIZE do
    local batch_img, batch_gt, batch_sizes = dv:next(BATCH_SIZE)
    if use_gpu then
      batch_img = batch_img:cuda()
    end

    local output = model:forward(batch_img)
    local sizes = {}
    local seq_len = output:size()[1] / BATCH_SIZE
    
    for i=1,BATCH_SIZE do 
      table.insert(sizes, seq_len) 
    end

    local grad_output = output:clone():zero()
    local loss = 0
    
    -- Compute loss function and gradients respect the output
    if use_gpu then
    loss = table.reduce(gpu_ctc(output, grad_output, batch_gt, sizes),
                          operator.add, 0)
    else
      output = output:float()
      grad_output = grad_output:float()
      loss = table.reduce(cpu_ctc(output, grad_output, batch_gt, sizes),
                            operator.add, 0)
    end
    
    -- Make loss function (and output gradients) independent of batch size
    -- and sequence length.
    loss = loss / (BATCH_SIZE * seq_len)
    valid_loss_epoch = valid_loss_epoch + BATCH_SIZE * loss
    xlua.progress(batch + BATCH_SIZE - 1, dv:numSamples())
  end

  train_loss_epoch = train_loss_epoch / dt:numSamples()
  valid_loss_epoch = valid_loss_epoch / dv:numSamples()
  --local gmin, gmax, mass = torch.sumarizeMagnitudes(
  --gradParameters, 0.85, 100)
  --print(string.format('Epoch = %-5d  Avg. Train Loss = %7.4f  -- ' ..
  -- '%5.2f%% of gradients are in range (%g, %g]',
  --epoch, train_loss_epoch, mass * 100, gmin, gmax))

  -- Write the results in the file
  file = io.open('../res/' .. id .. '.res', 'a')
  file:write(string.format('Epoch = %-5d  Avg. Train Loss = %7.4f  ' ..
      'Avg. Valid Loss = %7.4f\n',
         epoch, train_loss_epoch, valid_loss_epoch))
  file:close()

  -- Calculate if this is the best model so far
  if valid_loss_epoch < best_valid_loss_epoch then
    best_valid_loss_epoch = valid_loss_epoch
    -- Save model
    collectgarbage()
    torch.save(string.format('../model/' .. id .. '.t7', epoch), model:clearState())
  end
end
