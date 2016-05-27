--[[
Main entry point for training
]]--

------------------------------------
-- Includes
------------------------------------

require 'torch'
require 'warp_ctc'
require 'optim'
require 'xlua'
local cjson = require 'cjson'

require 'src.CurriculumBatcher'
require 'src.RandomBatcher'

--require 'src.Model-VGG_A'
require 'src.Model'
require 'src.utilities'
local opts = require 'train_opts'
local models = require 'models'

------------------------------------
-- Initializations
------------------------------------

local opt = opts.parse(arg)
print('Hyperparameters: ', opt)

task_id = os.date("%d.%m.%H.%M")

-- start log file
local file = io.open(opt.output_path .. '/' .. task_id .. '.csv', 'w')
local output_log_line = string.format('EPOCH BEST LOSS_TRAIN LOSS_VALID CER_TRAIN CER_VALID\n')
file:write(output_log_line)
file:close()

-- serialize a json file that has all the opts
cjson.encode_number_precision(4) -- number of sig digits to use in encoding
cjson.encode_sparse_array(true, 2, 10)
local json_opt = cjson.encode(opt)
local file = io.open(opt.output_path .. '/' .. task_id .. '.json', 'w')
file:write(json_opt)
file:close()

torch.manualSeed(opt.seed)
math.randomseed(opt.seed)

local rmsprop_opts = {
   learningRate = opt.learning_rate,
   alpha = opt.alpha,
}

if opt.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'

  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1) -- +1 because lua is 1-indexed

  if opt.backend == 'cudnn' then
    require 'cudnn'
  end
end

-- initialize data loaders
local dt = CurriculumBatcher(opt.training, true)
local dv = RandomBatcher(opt.validation, true)

-- initialize the model object
local model = models.setup(opt)

-- get the parameters vector
local parameters, gradParameters = model:getParameters()
print('Number of parameters: ', gradParameters:nElement())

------------------------------------
-- Main loop
------------------------------------

local best_valid_cer = nil
local epoch = 0

while true do
  -- Apply learning rate decay
  if epoch > opt.learning_rate_decay_after then
    rmsprop_opts.learningRate =
      rmsprop_opts.learningRate * opt.learning_rate_decay
  end
  -- Curriculum learning
  if epoch <= opt.curriculum_learning_after then
    dt:sample(opt.curriculum_learning_after *
              (1.0 - (epoch - 1) / opt.curriculum_learning_update), 5)
  else
    dt:sample(0, 5)
  end

  model:training()

  local train_loss_epoch = 0.0
  local train_num_edit_ops = 0
  local train_ref_length = 0

  for batch=1,dt:numSamples(),opt.batch_size do
    local batch_img, batch_gt, batch_sizes = dt:next(opt.batch_size)
    
    if opt.gpu >= 0 then
     batch_img = batch_img:cuda()
    end

    local feval = function(x)
      assert (x == parameters)
      --collectgarbage()
      gradParameters:zero()

      model:forward(batch_img)
      
      local output = model.output
      local sizes = {}
      local seq_len = output:size()[1] / opt.batch_size
      for i=1,opt.batch_size do table.insert(sizes, seq_len) end

      local grad_output = output:clone():zero()
      local loss = 0
      
      -- Compute loss function and gradients respect the output
      if opt.gpu >= 0 then
        loss = table.reduce(gpu_ctc(output, grad_output, batch_gt, sizes),
                              operator.add, 0)
      else
        output:float()
        grad_output:float()
        loss = table.reduce(cpu_ctc(output, grad_output, batch_gt, sizes),
                              operator.add, 0)
      end
      
      -- Perform framewise decoding to estimate CER
      local batch_decode = framewise_decode(opt.batch_size, output)
      for i=1,opt.batch_size do
        local num_edit_ops, _ = levenshtein(batch_decode[i], batch_gt[i])
        train_num_edit_ops = train_num_edit_ops + num_edit_ops
        train_ref_length = train_ref_length + #batch_gt[i]
      end

      -- Make loss function (and output gradients) independent of batch size
      -- and sequence length.
      loss = loss / (opt.batch_size * seq_len)
      grad_output:div(opt.batch_size * seq_len)
      
      -- Compute gradients of the loss function w.r.t parameters
      model:backward(batch_img, grad_output)
      
      -- L1 Normalization
      gradParameters:add(opt.weight_l1_decay, torch.sign(parameters))

      -- L2 Normalization
      gradParameters:add(opt.weight_l2_decay, parameters)
     
      -- Clip gradients
      if opt.grad_clip > 0 then
        gradParameters:clamp(-opt.grad_clip, opt.grad_clip)
      end
      
      train_loss_epoch = train_loss_epoch + opt.batch_size * loss
      return loss, gradParameters
    end

    optim.rmsprop(feval, parameters, rmsprop_opts)
    xlua.progress(batch + opt.batch_size - 1, dt:numSamples())
  end

  -- VALIDATION
  model:evaluate()
  dv:shuffle()

  local valid_loss_epoch = 0.0
  local valid_num_edit_ops = 0
  local valid_ref_length = 0
  
  for batch=1,dv:numSamples(),opt.batch_size do
    local batch_img, batch_gt, batch_sizes = dv:next(opt.batch_size)
    if opt.gpu >= 0 then
      batch_img = batch_img:cuda()
    end

    model:forward(batch_img)

    local output = model.output
    local sizes = {}
    local seq_len = output:size()[1] / opt.batch_size
    
    for i=1,opt.batch_size do 
      table.insert(sizes, seq_len) 
    end

    local grad_output = output:clone():zero()
    local loss = 0
    
    -- Compute loss function and gradients respect the output
    if opt.gpu >= 0 then
    loss = table.reduce(gpu_ctc(output, grad_output, batch_gt, sizes),
                          operator.add, 0)
    else
      output:float()
      grad_output = grad_output:float()
      loss = table.reduce(cpu_ctc(output, grad_output, batch_gt, sizes),
                            operator.add, 0)
    end
    
    -- Perform framewise decoding to estimate CER
    -- get rid of last layer
    local batch_decode = framewise_decode(opt.batch_size, output)
    for i=1,opt.batch_size do
      local num_edit_ops, _ = levenshtein(batch_gt[i], batch_decode[i])
      valid_num_edit_ops = valid_num_edit_ops + num_edit_ops
      valid_ref_length = valid_ref_length + #batch_gt[i]
    end

    -- Make loss function (and output gradients) independent of batch size
    -- and sequence length.
    loss = loss / (opt.batch_size * seq_len)
    valid_loss_epoch = valid_loss_epoch + opt.batch_size * loss
    xlua.progress(batch + opt.batch_size - 1, dv:numSamples())
  end

  train_loss_epoch = train_loss_epoch / dt:numSamples()
  valid_loss_epoch = valid_loss_epoch / dv:numSamples()

  train_cer_epoch = train_num_edit_ops / train_ref_length
  valid_cer_epoch = valid_num_edit_ops / valid_ref_length

  ------------------------------------
  -- Logging code
  ------------------------------------

  best_model = false 
  -- Calculate if this is the best checkpoint so far
  if best_valid_cer == nil or valid_cer_epoch < best_valid_cer then 
    best_valid_cer = valid_cer_epoch
    best_model = true

    local checkpoint = {}
    checkpoint.opt  = opt
    checkpoint.epoch = epoch
    checkpoint.model = model
  
    model:clearState()

    -- Only save t7 checkpoint if there is an improvement in CER
    torch.save(string.format(opt.output_path .. '/' .. task_id .. '.t7'), checkpoint)
  end

  -- Write the results in the file
  local file = io.open(opt.output_path .. '/' .. task_id .. '.csv', 'a')
  if best_model then
    output_log_line = string.format('%-5d   *  %10.6f %10.6f %9.2f %9.2f\n', 
            epoch, train_loss_epoch, valid_loss_epoch, train_cer_epoch * 100, valid_cer_epoch * 100)
  else
    output_log_line = string.format('%-5d      %10.6f %10.6f %9.2f %9.2f\n', 
            epoch, train_loss_epoch, valid_loss_epoch, train_cer_epoch * 100, valid_cer_epoch * 100)
  end
  file:write(output_log_line)
  file:close()

  -- stopping criterions
  epoch = epoch + 1
  -- Collect garbage every so often
  if epoch % 10 == 0 then collectgarbage() end
  if opt.max_epochs > 0 and epoch >= opt.max_epochs then break end
end
