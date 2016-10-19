#!/usr/bin/env th

--[[
Main entry point for training
]]--

------------------------------------
-- Includes
------------------------------------

-- TODO: Curriculum batcher does not work properly
-- require 'laia.CurriculumBatcher'; local Batcher = CurriculumBatcher
-- TODO: The type of batcher could be selected as an arg when calling train.lua
require 'laia'
require 'warp_ctc'
require 'optim'
require 'xlua'

local Batcher = laia.RandomBatcher

local opts = require 'laia.TrainOptions'


local sig = require 'posix.signal'
local exit_request = false
function handle_signal()
  exit_request = true
end
sig.signal(sig.SIGUSR1,handle_signal)

------------------------------------
-- Initializations
------------------------------------

local opt = opts.parse(arg)

-- First of all, update logging options and set random seeds
laia.log.loglevel = opt.log_level
laia.log.logfile  = opt.log_file
laia.log.logstderrthreshold = opt.log_stderr_threshold
laia.manualSeed(opt.seed)

-- Second, set cuda device
if opt.gpu >= 0 then
  cutorch.setDevice(opt.gpu + 1) -- +1 because lua is 1-indexed
end

-- Load initial checkpoint
local initial_checkpoint = torch.load(opt.model)
local model = nil
if torch.isTypeOf(initial_checkpoint, 'nn.Module') then
  assert(false, "TODO(jpuigcerver): This doesn't work anymore, since the " ..
         "train script assumes that the model was created with create_model")
  -- Load a torch nn.Module object
  model = initial_checkpoint
else
  -- TODO: Load a checkpoint, so we can continue training.
  -- This currently won't work, since we are not reading the options,
  -- like learningRate, from the checkpoint file.
  model = initial_checkpoint.model
end
assert(model ~= nil)
-- Place model to the correct device
if opt.gpu >= 0 then model:cuda() else model:float() end
-- Use cudnn implementation for all possible layers
if cudnn ~= nil then cudnn.convert(model, cudnn) end

-- Set RNG state for Laia
if initial_checkpoint.rng_state then
  laia.setRNGState(initial_checkpoint.rng_state)
end

-- Set current epoch from the checkpoint.
-- Note: We check that the initial checkpoint was not a nn.Module just in a
-- weird cas the module had some internal variable called epoch.
local epoch = (not torch.isTypeOf(initial_checkpoint, 'nn.Module') and
		 initial_checkpoint.epoch) or 0

-- Prepare HTML monitor
local monitor = nil
if opt.monitor_html ~= '' then
  monitor = laia.Monitor(
    opt.monitor_html, {
      loss = {
	name = 'Loss',
	xlabel = 'Epoch',
	ylabel = 'Loss',
	curves = {
	  train = {name = 'Train'},
	  valid = {name = 'Valid'}
	}
      },
      cer = {
	name = 'CER',
	xlabel = 'Epoch',
	ylabel = 'CER (%)',
	curves = {
	  train = {name = 'Train'},
	  valid = {name = 'Valid'}
	}
      }
  })
end

-- Configure RMSprop options
local rmsprop_opts = nil
if initial_checkpoint.rmsprop and not opt.force then
  rmsprop_opts = initial_checkpoint.rmsprop
else
  rmsprop_opts = {
    learningRate = opt.learning_rate,
    alpha = opt.alpha,
  }
end


-- Get the parameters and gradParameters vector
local parameters, gradParameters = model:getParameters()
laia.log.info('Number of parameters: ', gradParameters:nElement())

-- Determine the filename of the output model
local output_model_filename = opt.model
if opt.output_model ~= '' then
  output_model_filename = opt.output_model
end

local output_progress_file = nil
if opt.output_progress ~= '' then
  output_progress_file = io.open(opt.output_progress,
				 (epoch == 0 and 'w') or 'a')
  output_progress_file:write('# EPOCH   BEST?   TRAIN_LOSS   VALID_LOSS   TRAIN_CER   VALID_CER   TRAIN_TIME(min)   VALID_TIME(min)\n')
  output_progress_file:flush()
end

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

local dt = Batcher(opt)
dt:load(opt.training, opt.training_gt, opt.symbols_table)

local dv = Batcher(opt)
dv:load(opt.validation, opt.validation_gt, opt.symbols_table)

-- Check number of symbols and model output
-- TODO(jpuigcerver,mauvilsa): This assertion only works for the default
-- model created from create_model.lua, but this is not a good way of checking
-- the output size! For instance, the last layer may not have any parameters,
-- if a non-linear activation function is set after the linear layer.
--assert(dt:numSymbols()+1 == model:get(#model):parameters()[2]:size()[1],
--       string.format('Expected model output to have #symbols+1 dimensions! #symbols=%d vs. model=%d',
--                     dt:numSymbols(),model:get(#model):parameters()[2]:size()[1]))

-- TODO(jpuigcerver): Pass options to the image distorter
-- TODO(jpuigcerver): Image distorter only works for GPU!
local distorter = laia.ImageDistorter()

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
    local dc_i = batch_decode[i]
    local gt_i = batch_gt[i]
    if opt.cer_trim > 0 then
      dc_i = symbol_trim(dc_i, opt.cer_trim)
      gt_i = symbol_trim(gt_i, opt.cer_trim)
    end
    --local num_edit_ops, _ = levenshtein(batch_decode[i], batch_gt[i])
    local num_edit_ops, _ = levenshtein(dc_i, gt_i)
    batch_num_edit_ops = batch_num_edit_ops + num_edit_ops
    --batch_ref_length = batch_ref_length + #batch_gt[i]
    batch_ref_length = batch_ref_length + #gt_i
  end

  -- Make loss function (and output gradients) independent of batch size
  -- and sequence length.
  loss = loss / (opt.batch_size * seq_len)
  grad_output:div(opt.batch_size * seq_len)

  -- Compute gradients of the loss function w.r.t parameters
  if do_backprop then
    model:backward(batch_img, grad_output)
  end

  return loss, batch_num_edit_ops, batch_ref_length, batch_decode
end
-- END fb_pass

------------------------------------
-- Main loop
------------------------------------

-- This variable keeps track of the total number of times that the model
-- was updated.
local total_updates = 0

function train_epoch()
  --------------------------------
  --    TRAINING EPOCH START    --
  --------------------------------
  local train_time_start = os.time()
  dt:epochReset()
  train_loss_epoch = 0
  train_num_edit_ops = 0
  train_ref_length = 0
  for batch=1,train_num_samples,opt.batch_size do
    if exit_request then break end
    local batch_img, batch_gt, batch_sizes, batch_ids = dt:next(opt.batch_size)
    if opt.gpu >= 0 then batch_img = batch_img:cuda() end
    batch_img = distorter:distort(batch_img)

    local train_batch = function(x)
      assert (x == parameters)
      --collectgarbage()

      -- Regular backpropagation pass
      local batch_loss, batch_num_edit_ops, batch_ref_length, batch_decode =
        fb_pass(batch_img, batch_gt, true)

      -- Update monitor snapshot
      total_updates = total_updates + 1
      if monitor ~= nil and opt.monitor_snapshot > 0 and
	total_updates % opt.monitor_snapshot == 0 then
	  laia.log.warn('Model snapshots are not fully functional yet')
	  -- TODO(jpuigcerver): This is extremely slow, probably due to
	  -- memory transfers between GPU and CPU. Look into this.
	  --monitor:updateSnapshot(batch_img, model, batch_decode, batch_gt)
	  --monitor:writeHTML()
      end

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
        local adv_loss, adv_edit_ops, adv_ref_len, _ =
	  fb_pass(adv_img, batch_gt, true)
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
	-- Number of clamped gradients, for monitoring purposes
	local ncg = torch.abs(gradParameters):gt(opt.grad_clip):sum()
	if ncg > 0 then
	  laia.log.debug(
	    string.format('%d (%.2f%%) gradients clamped to [-%g,%g]',
			  ncg, 100 * ncg / gradParameters:nElement(),
			  opt.grad_clip, opt.grad_clip))
	  gradParameters:clamp(-opt.grad_clip, opt.grad_clip)
	end
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
    if laia.stdout_isatty and opt.show_epoch_bar then
      xlua.progress(batch + opt.batch_size - 1, train_num_samples)
    end
  end
  train_loss_epoch = train_loss_epoch / train_num_samples
  train_cer_epoch = train_num_edit_ops / train_ref_length
  local train_time_end = os.time()
  ------------------------------
  --    TRAINING EPOCH END    --
  ------------------------------
  return os.difftime(train_time_end, train_time_start)
end

function valid_epoch()
  ----------------------------------
  --    VALIDATION EPOCH START    --
  ----------------------------------
  local valid_time_start = os.time()
  dv:epochReset()
  valid_loss_epoch = 0
  valid_num_edit_ops = 0
  valid_ref_length = 0
  for batch=1,valid_num_samples,opt.batch_size do
    local batch_img, batch_gt, batch_sizes = dv:next(opt.batch_size)
    if opt.gpu >= 0 then
      batch_img = batch_img:cuda()
    end
    -- Forward pass
    local batch_loss, batch_num_edit_ops, batch_ref_length, batch_decode =
      fb_pass(batch_img, batch_gt, false)
    -- Compute EPOCH (not batch) errors
    valid_loss_epoch = valid_loss_epoch + opt.batch_size * batch_loss
    valid_num_edit_ops = valid_num_edit_ops + batch_num_edit_ops
    valid_ref_length = valid_ref_length + batch_ref_length
    -- Show progress bar only if running on a tty
    if laia.stdout_isatty and opt.show_epoch_bar then
      xlua.progress(batch + opt.batch_size - 1, valid_num_samples)
    end
  end
  valid_loss_epoch = valid_loss_epoch / valid_num_samples
  valid_cer_epoch = valid_num_edit_ops / valid_ref_length
  local valid_time_end = os.time()
  --------------------------------
  --    VALIDATION EPOCH END    --
  --------------------------------
  return os.difftime(valid_time_end, valid_time_start)
end

while opt.max_epochs <= 0 or epoch < opt.max_epochs do
  -- Epoch starts at 0, when the model is created
  epoch = epoch + 1

  -- Apply learning rate decay
  if epoch > opt.learning_rate_decay_after then
    rmsprop_opts.learningRate =
      rmsprop_opts.learningRate * opt.learning_rate_decay
  end

  local train_time = train_epoch()
  if exit_request then break end
  local valid_time = valid_epoch()


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
    checkpoint.model_opt = initial_checkpoint.model_opt
    checkpoint.train_opt = opt       -- Original training options
    checkpoint.epoch     = epoch
    checkpoint.model     = model
    -- Current RNG state
    checkpoint.rng_state = laia.getRNGState()
    -- Current rmsprop options (i.e. current learning rate)
    checkpoint.rmsprop   = rmsprop_opts
    model:clearState()
    -- Only save t7 checkpoint if there is an improvement in CER
    torch.save(output_model_filename, checkpoint)
  end

  -- Print progress of the loss function, CER and running times
  if output_progress_file ~= nil then
    output_progress_file:write(
      string.format(
	'%-7d   %s   %10.6f   %10.6f   %9.2f   %9.2f   %15.2f   %15.2f\n',
	epoch, (best_model and '  *  ') or '     ',
	train_loss_epoch, valid_loss_epoch,
	train_cer_epoch * 100, valid_cer_epoch * 100,
	train_time / 60.0, valid_time / 60.0))
    output_progress_file:flush()
  else
    laia.log.info(string.format(
		    'Epoch = %d  Loss = %10.4f / %10.4f  CER = %5.2f / %5.2f',
		    epoch, train_loss_epoch, valid_loss_epoch,
		    train_cer_epoch * 100, valid_cer_epoch * 100))
  end

  -- Update monitor with epoch progress
  if monitor ~= nil then
    monitor:updatePlots{
      loss = { train = train_loss_epoch,
	       valid = valid_loss_epoch },
      cer  = { train = train_cer_epoch * 100,
	       valid = valid_cer_epoch * 100 }
    }
    monitor:writeHTML()
  end

  -- Collect garbage every so often
  if epoch % 10 == 0 then collectgarbage() end
  if opt.max_no_improv_epochs > 0 and
  epoch - last_signif_improv_epoch >= opt.max_no_improv_epochs then break end
end

if output_progress_file ~= nil then output_progress_file:close() end
