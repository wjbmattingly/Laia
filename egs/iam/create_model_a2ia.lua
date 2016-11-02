#!/usr/bin/env th

require 'laia'

assert(cudnn ~= nil, 'create_model_a2ia.lua requires cudnn')

local cmd = torch.CmdLine('Create a DCNN-RNN model.')

cmd:text('Convolutional layer options:')
cmd:option('-cnn_dropout', '0',
	   'Dropout probability to the input of each convolutional layer')
cmd:option('-cnn_num_features', '6 20',
	   'Number of feature maps of the convolutional layers')
cmd:option('-cnn_kernel_size', '2,4 2,4',
	   'Kernel size of the convolutional layers')
cmd:option('-cnn_type', 'tanh',
	   'Type of the activation in each convolutional layer (values: ' ..
	   'tanh, relu, prelu, rrelu, leakyrelu, softplus)')
cmd:text()

cmd:text('Recurrent layer options:')
cmd:option('-lstm_dropout', '0', 'Dropout at the input of each LSTM layer')
cmd:option('-lstm_num_features', '2 10 50',
	   'Number of units of each LSTM layer')
cmd:text()

cmd:text('Other options:')
cmd:option('-seed', 0x012345, 'Seed for random numbers generation')
cmd:option('-linear_dropout', 0,
	   'Dropout probability to the input of the final linear layer')
cmd:text()

cmd:text('Arguments:')
cmd:argument('input_channels', 'Number of channels of the input images')
cmd:argument('input_height', 'Height of the input images')
cmd:argument('output_size',
	     'Number of output symbols (if you are going to use the CTC ' ..
             'loss include one additional element!)')
cmd:argument('output_file', 'Output file to store the model')
cmd:text()

local opt = cmd:parse(arg or {})
laia.log.info('Model hyperparameters: ', opt)

local cnn_num_features = string.split(opt.cnn_num_features)
local cnn_kernel_size = string.split(opt.cnn_kernel_size)
local cnn_type = string.split(opt.cnn_type)
local cnn_dropout = string.split(opt.cnn_dropout)
local lstm_num_features = string.split(opt.lstm_num_features)
local lstm_dropout = string.split(opt.lstm_dropout)

opt.input_channels = tonumber(opt.input_channels)
opt.input_height = tonumber(opt.input_height)
opt.output_size = tonumber(opt.output_size)
opt.seed = tonumber(opt.seed)

-- Initialize random seeds
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

-- Check that the specified parameters for the CNN layers make sense
-- Number of lstm+conv layers
local lstm_cnn_layers = #cnn_num_features
assert(lstm_cnn_layers > 0,
       'You must specifiy at least one cnn layer with -cnn_num_features')
assert(#cnn_kernel_size > 0, 'You must specify at least one kernel size')
assert(#cnn_type > 0, 'You must specify at least one activation function')
assert(#cnn_dropout > 0, 'You must specify at least one dropout probability')
assert(#cnn_kernel_size <= lstm_cnn_layers,
       'You specified more kernel sizes than convolutional layers!')
assert(#cnn_type <= lstm_cnn_layers,
       'You specified more activation types than convolutional layers!')
assert(#cnn_dropout <= lstm_cnn_layers,
       'You specified more dropout values than convolutional layers!')

-- Ensure that all options for the convolutional layers have the same
-- size (equal to the number of specified layers). The last option in a list
-- is copied to extend the list until a size of cnn_layers is achieved.
table.append_last(cnn_kernel_size, lstm_cnn_layers - #cnn_kernel_size)
table.append_last(cnn_type, lstm_cnn_layers - #cnn_type)
table.append_last(cnn_dropout, lstm_cnn_layers - #cnn_dropout)

-- Convert lists of strings to appropiate types and sizes
cnn_dropout = table.map(cnn_dropout, tonumber)
cnn_num_features = table.map(cnn_num_features, tonumber)
cnn_kernel_size = table.map(cnn_kernel_size, function(x)
  -- Each element in the kernel sizes list must be a pair of integers
  local t = table.map(string.split(x, '[^,]+'), tonumber)
  table.append_last(t, 2 - #t)
  return t
end)


-- Check that the specified parameters for the LSTM layers make sense
-- Number of lstm+linear layers
local lstm_layers = #lstm_num_features
local lstm_linear_layers = lstm_layers - #cnn_num_features
assert(lstm_linear_layers == 1,
       string.format('You must specify %d lstm layers with -lstm_num_features',
		     lstm_cnn_layers + 1))
assert(#lstm_dropout > 0,
       'You must specify at least one dropout probability')
assert(#lstm_dropout <= lstm_layers,
       'You specified more dropout values than lstm layers')
table.append_last(lstm_dropout, lstm_layers - #lstm_dropout)
lstm_dropout = table.map(lstm_dropout, tonumber)
lstm_num_features = table.map(lstm_num_features, tonumber)


function lstm_conv_block(input_depth, lstm_depth, cnn_depth,
			 kernel_w, kernel_h,
			 step_w, step_h,
			 activation, lstm_dropout, cnn_dropout)
  assert(input_depth > 0)
  assert(lstm_depth > 0)
  assert(cnn_depth > 0)
  assert(kernel_w > 0)
  assert(kernel_h > 0)
  activation = activation or 'tanh'
  lstm_dropout = lstm_dropout or 0
  cnn_dropout = cnn_dropout or 0
  step_w = step_w or kernel_w
  step_h = step_h or kernel_h
  local pad_w = (kernel_w - step_w) / 2
  local pad_h = (kernel_h - step_h) / 2

  local block = nn.Sequential()

  -- 4D-like LSTM layer
  block:add(laia.MDRNN(input_depth, lstm_depth, 'lstm', lstm_dropout))

  -- Process each direction with a different convolutional layer
  local conv_blocks = nn.ConcatTable()
  for i=0,3 do
    local cblock = nn.Sequential()
    cblock:add(nn.Narrow(2, 1 + i * lstm_depth, lstm_depth))
    if cnn_dropout > 0 and cnn_dropout < 1 then
      cblock:add(nn.SpatialDropout(cnn_dropout))
    end
    cblock:add(cudnn.SpatialConvolution(lstm_depth, cnn_depth,
					kernel_w, kernel_h,
					step_w, step_h,
					pad_w, pad_h))
    conv_blocks:add(cblock)
  end
  block:add(conv_blocks)

  -- Sum activations from each direction
  block:add(nn.CAddTable())

  -- Activation function
  if activation == 'relu' then
    block:add(cudnn.ReLU(true))
  elseif activation == 'tanh' then
    block:add(cudnn.Tanh())
  elseif activation == 'leakyrelu' then
    block:add(nn.LeakyReLU(true))
  elseif activation == 'softplus' then
    block:add(nn.SoftPlus())
  elseif activation == 'prelu' then
    block:add(nn.PReLU())
  elseif activation == 'rrelu' then
    block:add(nn.RReLU(1.0 / 8.0, 1.0 / 3.0, true))
  elseif activation == 'none' then
    -- do not add any activation
  else
    assert(false, string.format('Unknown activation function %s', activation))
  end

  return block
end

function lstm_linear_block(input_depth, lstm_depth, linear_depth,
			   activation, lstm_dropout, linear_dropout)
  activation = activation or 'none'
  lstm_dropout = lstm_dropout or 0
  linear_dropout = linear_dropout or 0

  local block = nn.Sequential()

  -- 4D-like LSTM layer
  block:add(laia.nn.MDRNN(input_depth, lstm_depth, 'lstm', lstm_dropout))

  -- Process each direction with a different linear layer
  local linear_blocks = nn.ConcatTable()
  for i=0,3 do
    local lblock = nn.Sequential()
    lblock:add(nn.Narrow(2, 1 + i * lstm_depth, lstm_depth))
    lblock:add(laia.nn.ImageColumnSequence())
    lblock:add(nn.Reshape(-1, lstm_depth, false))
    if linear_dropout > 0 and linear_droput < 1 then
      lblock:add(nn.SpatialDropout(linear_dropout))
    end
    lblock:add(nn.Linear(lstm_depth, linear_depth))
    linear_blocks:add(lblock)
  end
  block:add(linear_blocks)

  -- Sum activations from each direction
  block:add(nn.CAddTable())

  -- Activation function
  if activation == 'relu' then
    block:add(cudnn.ReLU(true))
  elseif activation == 'tanh' then
    block:add(cudnn.Tanh())
  elseif activation == 'leakyrelu' then
    block:add(nn.LeakyReLU(true))
  elseif activation == 'softplus' then
    block:add(nn.SoftPlus())
  elseif activation == 'prelu' then
    block:add(nn.PReLU())
  elseif activation == 'rrelu' then
    block:add(nn.RReLU(1.0 / 8.0, 1.0 / 3.0, true))
  elseif activation == 'none' then
    -- do not add any activation
  else
    assert(false, string.format('Unknown activation function %s', activation))
  end

  return block
end

local model = nn.Sequential()
-- Used to compute the depth of the images after all the convolutions
local curr_c = opt.input_channels
-- Append lstm + convolutional blocks
for i=1,lstm_cnn_layers do
  model:add(lstm_conv_block(curr_c, lstm_num_features[i], cnn_num_features[i],
			    cnn_kernel_size[i][1], cnn_kernel_size[i][2],
			    cnn_kernel_size[i][1], cnn_kernel_size[i][2],
			    cnn_type[i], lstm_dropout[i], cnn_dropout[i]))
  curr_c = cnn_num_features[i]
end
-- Append lstm + linear block
model:add(lstm_linear_block(curr_c, lstm_num_features[#lstm_num_features],
			    opt.output_size, 'none',
			    lstm_dropout[#lstm_dropout], opt.linear_dropout))

-- Save model to disk
local checkpoint = {
  model = model,
  model_opt = opt
}
torch.save(opt.output_file, checkpoint)
