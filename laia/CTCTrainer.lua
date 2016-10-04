require 'warp_ctc'
local xlua = wrequire 'xlua'

local CTCTrainer, parent = torch.class('laia.CTCTrainer')

-- Basic usage example:
-- trainer = CTCTrainer(model, train_batcher, valid_batcher, optim.rmsprop)
-- trainer:start()
-- while true do
--   train_epoch_info = trainer:trainEpoch(rmsprop_opts)
--   valid_epoch_info = trainer:validEpoch()
--   print(train_epoch_info.loss, valid_epoch_info.loss)
-- end

function CTCTrainer:__init(model, train_batcher, valid_batcher, optimizer)
  self._model = model
  self._train_batcher = train_batcher
  self._valid_batcher = valid_batcher
  self._optimizer = optimizer
  -- Training options
  self._opt = {
    batch_size = 16,
    use_distortions = false,
    cer_trim = 0,
    snapshot_interval = 0,
    display_progress_bar = true
  }
  -- Use :start() to initialize the trainer
  self._initialized = false
end

function CTCTrainer:setModel(model)
  self._model = model
  self._initialized = false
end

function CTCTrainer:setTrainBatcher(batcher)
  self._train_batcher = batcher
  self._initialized = false
end

function CTCTrainer:setValidBatcher(batcher)
  self._valid_batcher = batcher
  self._initialized = false
end

function CTCTrainer:setOptimizer(optimizer)
  self._optimizer = optimizer
  self._initialized = false
end

function CTCTrainer:setDistorter(distorter)
  self._distorter = distorter
  self._initialized = false
end

function CTCTrainer:setAdversarialRegularizer(regularizer)
  self._adversarial_regularizer = regularizer
  self._initialized = false
end

function CTCTrainer:setWeightRegularizer(regularizer)
  self._weight_regularizer = regularizer
  self._initialized = false
end

function CTCTrainer:registerOptions(parser)
  parser:option('-b --batch_size', 'Batch size', 16, tonumber)
  parser:option('--use_distortions',
		'If true, augment the training set using random distortions',
		true, toboolean)
  parser:option('--cer_trim',
		'For computing CER, removes leading, trailing and ' ..
                'repetitions of given symbol number (i.e. space)', 0, tonumber)
  parser:option('--snapshot_interval',
		'If n>0, create a snapshot of the model every n batches',
		0, tonumber):argname('<n>')
  parser:option('--grad_clip',
		'If c>0, clip gradients to the range [-c,+c]',
		0, tonumber):argname('<c>')
  parser:option('--display_progress_bar',
		'Display a progress bar on the terminal showing the status ' ..
		'of the training and validation epoch. Note: if you ' ..
		'redirect the output to a file, the progress bar is not ' ..
                'displayed', true, toboolean)
end

function CTCTrainer:setOptions(opts)
  table.update_values(self._opt, opts)
  self:checkOptions()
end

function CTCTrainer:checkOptions()
  assert(self._opt.batch_size > 0 and isint(self._opt.batch_size),
	 ('Batch size must be positive integer (value = %s)'):format(
	   self._opt.batch_size))
  assert(isint(self._opt.cer_trim),
	 ('CER trim symbol must be an integer (value = %s)'):format(
	   self._opt.cer_trim))
  assert(isint(self._opt.snapshot_interval),
	 ('Snapshot interval must be an integer (value = %s)'):format(
	   self._opt.snapshot_interval))
  assert(type(self._opt.grad_clip) == 'number',
	 ('Gradient clip value must be a number (value = %s)'):format(
	   self._opt.grad_clip))
  assert(type(self._opt.display_progress_bar) == 'boolean',
	 ('Display progress bar must be a boolean (value = %s)'):format(
	   self._opt.display_progress_bar))
  -- Log some warnings
  if self._opt.display_progress_bar and not xlua then
    laia.log.warn('Progress bar not displayed, xlua was not found')
  end
  if self._opt.display_progress_bar and not laia.stdout_isatty then
    laia.log.warn('Progress bar not displayed, stdout is redirected')
  end
end

function CTCTrainer:start()
  assert(self._model ~= nil)
  assert(self._train_batcher ~= nil and self._train_batcher:numSamples() > 0)
  assert(self._valid_batcher ~= nil)
  assert(self._optimizer ~= nil)
  assert(self._distorter or not self._use_distortions)

  -- Flatten the model parameters into a single big chunk of memory.
  self._parameters, self._gradParameters = self._model:getParameters()
  -- Define _gradOutput that will be used for different batches to avoid
  -- multiple data allocation/dellocation.
  self._gradOutput = torch.Tensor()
  -- Total number of precessed training and validation batches, used to update
  -- the monitor snapshot only at certain times.
  self._num_processed_train_batches = 0
  self._num_processed_valid_batches = 0
  -- Compute the number of training samples to process in each epoch.
  -- Note: This number may be different from the total number of training
  -- samples, or the value of --num_epoch_samples, because of the batch_size.
  if self._opt.num_epoch_samples > 0 then
    self._train_num_samples = self._opt.batch_size *
      math.ceil(self._opt.num_epoch_samples / self._opt.batch_size)
  else
    self._train_num_samples = self._opt.batch_size *
      math.ceil(self._train_batcher:numSamples() / self._opt.batch_size)
  end

  -- Compute the number of validation samples to process in each epoch.
  -- See Note before.
  self._valid_num_samples = self._opt.batch_size *
    math.ceil(self._valid_batcher:numSamples() / self._opt.batch_size)

  self._initialized = true
end

function CTCTrainer.exitRequested()
  return CTCTrainer._exit_request
end

function CTCTrainer:trainEpoch(optimizer_params, batcher_reset_params)
  assert(self._initialized, 'CTCTrainer must be initialized with :start()')
  -- Reset batcher with the given parameters
  self._train_batcher:epochReset(batcher_reset_params)
  -- Useful information for monitoring the performance on trainining data
  local epoch_info = {
    loss        = 0,
    posteriors  = {},
    num_ins_ops = {},
    num_del_ops = {},
    num_sub_ops = {},
    hyp_trim    = {},
    ref_trim    = {},
    time_start  = os.time(),
    time_end    = nil
  }
  self._resetCosts(epoch_info)
  for b=1,self._train_num_samples,self._opt.batch_size do
    -- If exit signal was captured, terminate
    if CTCTrainer._exit_request then return nil end
    -- Load batch from batcher
    local batch_img, batch_gt, batch_sizes =
      self._train_batcher:next(self._opt.batch_size)
    -- Ensure that batch is in the same device as the model
    batch_img = batch_img:type(self._model.type())
    -- Apply distortions, if a distorter was given
    if self._distorter and self._opt.use_distortions then
      if batch_img:type() == 'torch.CudaTensor' then
	batch_img = self._distorter:distort(batch_img)
      else
	laia.log.error('Data distortions are only implemented on the GPU. ' ..
                       'If you badly need distortions running on the CPU, ' ..
                       'report a new issue to ' ..
                       'https://github.com/jpuigcerver/imgdistort/issues')
      end
    end
    -- Run optimizer on the batch
    self._optimizer(
      function(_)
	local batch_costs = self:_trainBatch(batch_img, batch_gt)
	CTCTrainer._updateCosts(epoch_info, batch_costs)
	return batch_costs.loss, self._gradParameters
      end, self._parameters, optimizer_params)
    -- Show progress bar only if running on a tty
    if xlua and laia.stdout_isatty and self._opt.display_progress_bar then
      xlua.progress(batch + self._opt.batch_size - 1, self._train_num_samples)
    end
    -- Update number of processed batches
    self._num_processed_train_batches = self._num_processed_train_batches + 1
  end
  epoch_info.time_end = os.time()
  return epoch_info
end

function CTCTrainer:validEpoch(batcher_reset_params)
  assert(self._initialized, 'CTCTrainer must be initialized with :start()')
  -- Reset batcher with the given parameters
  self._valid_batcher:epochReset(batcher_reset_params)
  -- Useful information for monitoring the performance on validation data
  local epoch_info = {
    loss        = 0,
    posteriors  = {},
    num_ins_ops = {},
    num_del_ops = {},
    num_sub_ops = {},
    hyp_trim    = {},
    ref_trim    = {},
    time_start  = os.time(),
    time_end    = nil
  }
  self._resetCosts(epoch_info)
  for b=1,self._valid_num_samples,self._opt.batch_size do
    -- If exit signal was captured, terminate
    if CTCTrainer._exit_request then return nil end
    -- Load batch from batcher
    local batch_img, batch_gt, batch_sizes =
      self._train_batcher:next(self._opt.batch_size)
    -- Ensure that batch is in the same device as the model
    batch_img = batch_img:type(self._model.type())
    -- Forward pass
    local batch_costs = self:_fbPass(batch_img, batch_gt, false)
    CTCTrainer._updateCosts(epoch_info, batch_costs)
    -- Show progress bar only if running on a tty
    if xlua and laia.stdout_isatty and self._opt.display_progress_bar then
      xlua.progress(batch + self._opt.batch_size - 1, self._valid_num_samples)
    end
    -- Update number of processed batches
    self._num_processed_valid_batches = self._num_processed_valid_batches + 1
  end
  epoch_info.time_end = os.time()
  return epoch_info
end

-- Perform forward/backward on a training batch, and apply the regularizers
-- to obtain the regularized loss function and the gradient of it w.r.t. the
-- model parameters.
-- Note: This updates self._gradParameters directly and returs the different
-- costs on the training batch for monitoring purposes. See the call to
-- optimizer() in trainEpoch() to see how this method is used.
function CTCTrainer:_trainBatch(batch_img, batch_gt)
  -- Regular backpropagation pass
  local batch_costs = self:_fbPass(batch_img, batch_gt, true)

  -- Adversarial samples regularization
  if self._adversarial_regularizer then
    batch_costs.loss = adversarial_regularizer:regularize(
      batch_costs.loss, self._model, batch_img,
      function(x) return self:_fbPass(x, batch_gt, true).loss end)
  end

  -- Weight decay regularization
  if self._weight_regularizer then
    batch_costs.loss = weight_regularizer:regularize(
      batch_costs.loss, self._model)
  end

  -- Clip gradients
  if self._opt.grad_clip > 0 then
    -- Number of clamped gradients, for debugging purposes
    local ncg = torch.abs(self._gradParameters):gt(self._opt.grad_clip):sum()
    if ncg > 0 then
      laia.log.debug(('%d (%.2f%%) gradients clamped to [-%g,%g]'):format(
	  ncg, 100 * ncg / gradParameters:nElement(),
	  self._opt.grad_clip, self._opt.grad_clip))
      self._gradParameters:clamp(-self._opt.grad_clip, self._opt.grad_clip)
    end
  end

  return batch_costs
end

-- Perform forward (and optionally, backprop) pass. This is common
-- code used for both training and evaluation of the model.
function CTCTrainer:_fbPass(batch_img, batch_gt, do_backprop)
  do_backprop = do_backprop or false
  if do_backprop then
    self._model:training()
    self._model:zeroGradParameters()
  else
    self._model:evaluate()
  end
  local batch_size = batch_img:size(1)
  assert(batch_size == #batch_gt,
	 ('The number of transcripts is not equal to the number of images '..
	  '(expected = %d, actual = %d)'):format(batch_size, #batch_gt))

  -- Forward pass
  self._model:forward(batch_img)
  local output = self._model.output
  -- Set _gradOutput to have the same size as the output and fill it with zeros
  self._gradOutput = self._gradOutput:typeAs(output):resizeAs(output):zero()

  -- TODO(jpuigcerver): This assumes that all sequences have the same number
  -- of frames, which should not be the case, since padding should be ignored!
  local sizes = {}
  local seq_len = output:size()[1] / batch_size
  for i=1,batch_size do table.insert(sizes, seq_len) end

  -- Compute loss function and gradients w.r.t. the output
  local batch_losses = nil
  if self._model:type() == 'torch.CudaTensor' then
    batch_losses = gpu_ctc(output, self._gradOutput, batch_gt, sizes)
  elseif self._model:type() == 'torch.FloatTensor' then
    batch_losses = cpu_ctc(output, self._gradOutput, batch_gt, sizes)
  else
    laia.log.fatal(
      ('CTC is not implemented for tensors of type %q'):format(output:type()))
  end

  -- Perform framewise decoding to estimate CER
  local batch_decode = framewise_decode(batch_size, output)
  local batch_dc_trim, batch_gt_trim = {}, {}
  local batch_num_ins_ops, batch_num_del_ops, batch_num_sub_ops = {}, {}, {}
  for i=1,#batch_decode do
    local dc_i = batch_decode[i]
    local gt_i = batch_gt[i]
    if self._cer_trim > 0 then
      dc_i = symbol_trim(dc_i, self._cer_trim)
      gt_i = symbol_trim(gt_i, self._cer_trim)
    end
    local _, edit_ops = levenshtein(dc_i, gt_i)
    table.insert(batch_num_ins_ops, edit_ops.ins)
    table.insert(batch_num_del_ops, edit_ops.del)
    table.insert(batch_num_sub_ops, edit_ops.sub)
    table.insert(batch_dc_trim, dc_i)
    table.insert(batch_gt_trim, gt_i)
  end

  -- Compute gradients of the loss function w.r.t parameters
  if do_backprop then
    self._model:backward(batch_img, self._gradOutput)
  end

  -- Make gradients independent of the batch size and sequence length
  self._gradParameters:div(self._gradOutput:size(1))

  -- Return, for each sample in the batch, the total loss (including
  -- regularization terms, adversarial, etc), the posterior probability
  -- of the reference sequence, number of edit operations of each type, the
  -- decoded (hypothesis) sequence and the reference sequence (after triming).
  return {
    -- Sum individual batch losses and normalize the batch loss taking into
    -- account the batch size and the length of the samples in the batch
    loss        =
      table.reduce(batch_losses, operator.add, 0) / self._gradOutput:size(1),
    -- Convert losses to (log-)posteriors, i.e loss = -log p(y|x)
    posteriors  = table.map(batch_losses, function(x) return -x end),
    num_ins_ops = batch_num_ins_ops,
    num_del_ops = batch_num_del_ops,
    num_sub_ops = batch_num_sub_ops,
    hyp_trim    = batch_dc_trim,
    ref_trim    = batch_gt_trim
  }
end

-- Usage:
-- a = { loss = 5, posteriors = {-0.2} }
-- b = { loss = 2, posteriors = {-0.01, -1.1} }
-- CTCTrainer._updateCosts(a, b)
-- print(a)
-- { loss = 7, posteriors = {-0.2, -0.01, -1.1} }
function CTCTrainer._updateCosts(dst, src)
  for k,v in pairs(src) do
    if type(v) == 'number' then
      assert(type(dst[k]) == 'number')
      dst[k] = dst[k] + v
    elseif type(v) == 'table' then
      assert(type(dst[k]) == 'table')
      table.foreach(v, function(x) table.insert(dst[k], x) end)
    end
  end
end

-- Usage:
-- a = { loss = 5, posteriors = {-0.2, -0.5} }
-- CTCTrainer._resetCosts(a)
-- print(a)
-- { loss = 0, posteriors = { } }
function CTCTrainer._resetCosts(dst)
  for k,v in pairs(dst) do
    if type(v) == 'number' then
      dst[k] = 0
    elseif type(v) == 'table' then
      dst[k] = {}
    end
  end
end

-- CTCTrainer is responsive to user signals to abort training in a graceful way
local sig = require 'posix.signal'
CTCTrainer._exit_request = false
function CTCTrainer.handle_signal()
  CTCTrainer._exit_request = true
end
sig.signal(sig.SIGINT, CTCTrainer.handle_signal)

return CTCTrainer
