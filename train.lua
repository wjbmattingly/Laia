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

-- TODO: Curriculum batcher does not work properly
-- require 'src.CurriculumBatcher'; local Batcher = CurriculumBatcher
-- TODO: The type of batcher could be selected as an arg when calling train.lua
require 'src.RandomBatcher'; local Batcher = RandomBatcher
require 'src.ImageDistorter'

require 'src.utilities'
local opts = require 'src.TrainOptions'

local term = require 'term'
local isatty = term.isatty(io.stdout)

------------------------------------
-- Initializations
------------------------------------

local opt = opts.parse(arg)
print('Training hyperparameters: ', opt)

-- RMSprop options
local rmsprop_opts = {
   learningRate = opt.learning_rate,
   alpha = opt.alpha,
}

torch.manualSeed(opt.seed)
if opt.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1) -- +1 because lua is 1-indexed
end


local initial_checkpoint = torch.load(opt.model)
local model = nil
local epoch = 0
if torch.isTypeOf(initial_checkpoint, 'nn.Module') then
   -- Load a torch nn.Module object
   model = initial_checkpoint
else
   -- TODO: Load a checkpoint, so we can continue training.
   -- This currently won't work, since we are not reading the options,
   -- like learningRate, from the checkpoint file.
   model = initial_checkpoint.model
   epoch = initial_checkpoint.epoch or 0
end
assert(model ~= nil)
assert(epoch ~= nil)

-- TODO(jpuigcerver): Allow model on CPU, or not using cudnn
model:cuda()
cudnn.convert(model, cudnn)

-- get the parameters vector
local parameters, gradParameters = model:getParameters()
print('Number of parameters: ', gradParameters:nElement())

-- Perform forward (and optionally, backprop) pass. This is common
-- code used for both training and evaluation of the model.
local fb_pass = function(batch_img, batch_gt, do_backprop)
  do_backprop = do_backprop or false
  if do_backprop then
    model:training()
  else
    model:evaluate()
  end

  gradParameters:zero()      -- Reset parameter gradients to 0
  model:forward(batch_img)   -- Forward pass

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
  local batch_num_edit_ops = 0
  local batch_ref_length = 0
  for i=1,opt.batch_size do
    local num_edit_ops, _ = levenshtein(batch_decode[i], batch_gt[i])
    batch_num_edit_ops = batch_num_edit_ops + num_edit_ops
    batch_ref_length = batch_ref_length + #batch_gt[i]
  end

  -- Make loss function (and output gradients) independent of batch size
  -- and sequence length.
  loss = loss / (opt.batch_size * seq_len)
  grad_output:div(opt.batch_size * seq_len)

  -- Compute gradients of the loss function w.r.t parameters
  if do_backprop then
    model:backward(batch_img, grad_output)
  end

  return loss, batch_num_edit_ops, batch_ref_length
end
-- END fb_pass

-- Determine the filename of the output model
local output_model_filename = opt.model
if opt.output_model ~= '' then
  output_model_filename = opt.output_model
end

local output_progress_file = nil
local progress_header = '# EPOCH   BEST?   TRAIN_LOSS   VALID_LOSS   TRAIN_CER   VALID_CER   TRAIN_TIME   VALID_TIME\n'
if opt.output_progress ~= '' then
  output_progress_file = io.open(opt.output_progress, 'w')
  output_progress_file:write(progress_header)
  output_progress_file:flush()
else
  io.stdout:write(progress_header)
  io.stdout:flush()
end

opt.channels = initial_checkpoint.model_opt.input_channels
opt.gt_file = opt.training_gt
local dt = Batcher(opt.training, opt)
opt.gt_file = opt.validation_gt
local dv = Batcher(opt.validation, opt)

-- Check number of symbols and model output
assert(dt:numSymbols()+1 == model:get(#model):parameters()[2]:size()[1],
       string.format('Expected model output to have #symbols+1 dimensions! #symbols=%d vs. model=%d',
                     dt:numSymbols(),model:get(#model):parameters()[2]:size()[1]))

-- TODO(jpuigcerver): Pass options to the image distorter
-- TODO(jpuigcerver): Image distorter only works for GPU!
local distorter = ImageDistorter()

-- Keep track of the performance on the training data
local train_loss_epoch = 0.0
local train_cer_epoch = 0.0
local train_num_edit_ops = 0
local train_ref_length = 0
local train_num_samples = nil
if opt.num_samples_epoch < 1 then
   train_num_samples =
      opt.batch_size * math.ceil(dt:numSamples() / opt.batch_size)
else
   train_num_samples =
      opt.batch_size * math.ceil(opt.num_samples_epoch / opt.batch_size)
end
-- Keep track of the performance on the validation data
local valid_loss_epoch = 0.0
local valid_cer_epoch = 0.0
local valid_num_edit_ops = 0
local valid_ref_length = 0
local valid_num_samples =
   opt.batch_size * math.ceil(dv:numSamples() / opt.batch_size)

local best_criterion_value = nil
local best_criterion_epoch = nil
local last_signif_improv_epoch = nil
local current_criterion_value = {
   train_loss = function() return train_loss_epoch end,
   train_cer = function() return train_cer_epoch end,
   valid_loss = function() return valid_loss_epoch end,
   valid_cer = function() return valid_cer_epoch end
}
assert(current_criterion_value[opt.early_stop_criterion] ~= nil,
       string.format('Early stop criterion %q is not supported!',
                     opt.early_stop_criterion))

------------------------------------
-- Main loop
------------------------------------

while opt.max_epochs <= 0 or epoch < opt.max_epochs do
  -- Epoch starts at 0, when the model is created
  epoch = epoch + 1

  -- Apply learning rate decay
  if epoch > opt.learning_rate_decay_after then
    rmsprop_opts.learningRate =
      rmsprop_opts.learningRate * opt.learning_rate_decay
  end

  --------------------------------
  --    TRAINING EPOCH START    --
  --------------------------------
  local train_time_start = os.time()
  dt:epochReset()
  train_loss_epoch = 0
  train_num_edit_ops = 0
  train_ref_length = 0
  for batch=1,train_num_samples,opt.batch_size do
    local batch_img, batch_gt, batch_sizes, batch_ids = dt:next(opt.batch_size)
    if opt.gpu >= 0 then
     batch_img = batch_img:cuda()
    end
     batch_img = distorter:distort(batch_img)

    local train_batch = function(x)
      assert (x == parameters)
      --collectgarbage()

      -- Regular backpropagation pass
      local batch_loss, batch_num_edit_ops, batch_ref_length =
        fb_pass(batch_img, batch_gt, true)

      -- Adversarial training
      if opt.adversarial_weight > 0.0 then
        local dJx = model:get(1).gradInput  -- Gradient w.r.t. the inputs
        -- Distort images so that the loss increases the maximum but the
        -- pixel-wise differences do not exceed opt.adversarial_epsilon
        local adv_img = torch.add(batch_img, opt.adversarial_epsilon,
                                  torch.sign(dJx))

        local orig_gradParameters = torch.mul(gradParameters,
                                              1.0 - opt.adversarial_weight)
        -- Backprop pass to get the gradients respect the adv images
        local adv_loss, adv_edit_ops, adv_ref_len = fb_pass(adv_img, batch_gt,
                                                            true)
        -- dJ_final = (1 - w) * dJ_orig + w * dJ_adv
        gradParameters:add(orig_gradParameters,
                           opt.adversarial_weight, gradParameters)
      end

      -- L1 Normalization
      gradParameters:add(opt.weight_l1_decay, torch.sign(parameters))

      -- L2 Normalization
      gradParameters:add(opt.weight_l2_decay, parameters)

      -- Clip gradients
      if opt.grad_clip > 0 then
        gradParameters:clamp(-opt.grad_clip, opt.grad_clip)
      end

      -- Compute EPOCH (not batch) errors
      train_loss_epoch = train_loss_epoch + opt.batch_size * batch_loss
      train_num_edit_ops = train_num_edit_ops + batch_num_edit_ops
      train_ref_length = train_ref_length + batch_ref_length

      return batch_loss, gradParameters
    end
    -- RMSprop on the batch
    optim.rmsprop(train_batch, parameters, rmsprop_opts)
    -- Show progress bar only if running on a tty
    if isatty then
      xlua.progress(batch + opt.batch_size - 1, train_num_samples)
    end
  end
  train_loss_epoch = train_loss_epoch / train_num_samples
  train_cer_epoch = train_num_edit_ops / train_ref_length
  local train_time_end = os.time()
  ------------------------------
  --    TRAINING EPOCH END    --
  ------------------------------


  ----------------------------------
  --    VALIDATION EPOCH START    --
  ----------------------------------
  local valid_time_start = os.time()
  dt:epochReset()
  valid_loss_epoch = 0
  valid_num_edit_ops = 0
  valid_ref_length = 0
  for batch=1,valid_num_samples,opt.batch_size do
    local batch_img, batch_gt, batch_sizes = dv:next(opt.batch_size)
    if opt.gpu >= 0 then
      batch_img = batch_img:cuda()
    end
    -- Forward pass
    local batch_loss, batch_num_edit_ops, batch_ref_length =
       fb_pass(batch_img, batch_gt, false)
    -- Compute EPOCH (not batch) errors
    valid_loss_epoch = valid_loss_epoch + opt.batch_size * batch_loss
    valid_num_edit_ops = valid_num_edit_ops + batch_num_edit_ops
    valid_ref_length = valid_ref_length + batch_ref_length
    -- Show progress bar only if running on a tty
    if isatty then
      xlua.progress(batch + opt.batch_size - 1, valid_num_samples)
    end
  end
  valid_loss_epoch = valid_loss_epoch / valid_num_samples
  valid_cer_epoch = valid_num_edit_ops / valid_ref_length
  local valid_time_end = os.time()
  --------------------------------
  --    VALIDATION EPOCH END    --
  --------------------------------

  local best_model = false
  local curr_crit_value = current_criterion_value[opt.early_stop_criterion]()
  if best_criterion_value == nil or curr_crit_value < best_criterion_value then
    best_model = true
    if best_criterion_value == nil or
      ((best_criterion_value - curr_crit_value) / best_criterion_value) >= opt.min_relative_improv then
      last_signif_improv_epoch = epoch
    end
  end

  if best_model then
    best_criterion_value = curr_crit_value
    best_criterion_epoch = epoch
    local checkpoint = {}
    checkpoint.train_opt = opt       -- Original training options
    checkpoint.epoch     = epoch
    checkpoint.model     = model
    model:clearState()
    -- Only save t7 checkpoint if there is an improvement in CER
    torch.save(output_model_filename, checkpoint)
  end

  -- Print progress of the loss function, CER and running times
  local progress_line = string.format(
     '%-7d   %s   %10.6f   %10.6f   %9.2f   %9.2f   %10.2f   %10.2f\n',
     epoch, (best_model and '  *  ') or '     ',
     train_loss_epoch, valid_loss_epoch,
     train_cer_epoch * 100, valid_cer_epoch * 100,
     os.difftime(train_time_end, train_time_start) / 60.0,
     os.difftime(valid_time_end, valid_time_start) / 60.0)
  if output_progress_file ~= nil then
    output_progress_file:write(progress_line)
    output_progress_file:flush()
  else
    io.stdout:write(progress_line)
    io.stdout:flush()
  end

  -- Collect garbage every so often
  if epoch % 10 == 0 then collectgarbage() end
  if opt.max_no_improv_epochs > 0 and
  epoch - last_signif_improv_epoch >= opt.max_no_improv_epochs then break end
end

if output_progress_file ~= nil then output_progress_file:close() end
