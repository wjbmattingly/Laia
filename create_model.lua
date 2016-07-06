require 'cudnn'
require 'src.utilities'

local cmd = torch.CmdLine('Create a DCNN-RNN model.')

cmd:text('Convolutional layer options:')
cmd:option('-cnn_kernel_size', '3,3 3,3 3,3 3,3',
	   'Kernel size of the convolutional layers')
cmd:option('-cnn_num_features', '16 16 32 32',
	   'Number of feature maps of the convolutional layers')
cmd:option('-cnn_maxpool_size', '2,2 2,2 0 2,2',
	   'Max pooling size after each convolutional layer (0 disables max ' ..
	   'pooling for that layer)')
cmd:option('-cnn_batch_norm', 'false false true false',
	   'Perform batch normalization before the activation function in ' ..
           'each convolutional layer')
cmd:option('-cnn_type', 'relu',
	   'Type of the activation in each convolutional layer (values: ' ..
	   'tanh, relu, prelu, rrelu, leakyrelu, softplus)')
cmd:option('-cnn_dropout', '0 0 0 0',
	   'Dropout probability to the input of each convolutional layer')
cmd:text()

cmd:text('Recurrent layer options:')
cmd:option('-rnn_dropout', 0.5,
	   'Dropout probability to the input of each recurrent layer')
cmd:option('-rnn_layers', 3, 'Number of recurrent layers')
cmd:option('-rnn_units', 256, 'Number of units in each recurrent layer')
cmd:option('-rnn_type', 'blstm',
	   'Type of the recurrent layers (values: blstm, bgru)')
cmd:text()

cmd:text('Other options:')
cmd:option('-seed', 0x012345, 'Seed for random numbers generation')
cmd:option('-linear_dropout', 0.5,
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

local cnn_num_features = string.split(opt.cnn_num_features)
local cnn_kernel_size = string.split(opt.cnn_kernel_size)
local cnn_maxpool_size = string.split(opt.cnn_maxpool_size)
local cnn_batch_norm = string.split(opt.cnn_batch_norm)
local cnn_type = string.split(opt.cnn_type)
local cnn_dropout = string.split(opt.cnn_dropout)
opt.input_channels = tonumber(opt.input_channels)
opt.input_height = tonumber(opt.input_height)
opt.output_size = tonumber(opt.output_size)
opt.rnn_dropout = tonumber(opt.rnn_dropout)
opt.rnn_layers = tonumber(opt.rnn_layers)
opt.rnn_units = tonumber(opt.rnn_units)
opt.seed = tonumber(opt.seed)

-- Number of conv layers determined by the number of elements in the
-- -cnn_num_features option.
local cnn_layers = #cnn_num_features

-- Check that the specified parameters make sense
assert(cnn_layers > 0, 'You must use at least one convolutional layer')
assert(#cnn_kernel_size > 0, 'You must specify at least one kernel size')
assert(#cnn_type > 0,
       'You must specify at least one activation function for the ' ..
       'convolutional layers')
if #cnn_maxpool_size == 0 then cnn_maxpool_size = { '0' } end
if #cnn_batch_norm == 0 then cnn_batch_norm = { 'false' } end
if #cnn_dropout == 0 then cnn_dropout = { '0' } end
assert(#cnn_kernel_size <= cnn_layers,
       'You specified more kernel sizes than convolutional layers!')
assert(#cnn_maxpool_size <= cnn_layers,
       'You specified more max pooling sizes than convolutional layers!')
assert(#cnn_batch_norm <= cnn_layers,
       'You specified more batch norm layers than convolutional layers!')
assert(#cnn_type <= cnn_layers,
       'You specified more activation types than convolutional layers!')
assert(#cnn_dropout <= cnn_layers,
       'You specified more dropout values than convolutional layers!')
-- Ensure that all options for the convolutional layers have the same
-- size (equal to the number of specified layers). The last option in a list
-- is copied to extend the list until a size of cnn_layers is achieved.
table.extend_with_last_element(cnn_kernel_size, cnn_layers)
table.extend_with_last_element(cnn_maxpool_size, cnn_layers)
table.extend_with_last_element(cnn_batch_norm, cnn_layers)
table.extend_with_last_element(cnn_type, cnn_layers)
table.extend_with_last_element(cnn_dropout, cnn_layers)
-- Convert lists of strings to appropiate types and sizes
cnn_dropout = table.map(cnn_dropout, tonumber)
cnn_num_features = table.map(cnn_num_features, tonumber)
cnn_kernel_size = table.map(cnn_kernel_size, function(x)
    -- Each element in the kernel sizes list must be a pair of integers
    local t = table.map(string.split(x, '[^,]+'), tonumber)
    table.extend_with_last_element(t, 2)
    return t
end)
cnn_maxpool_size = table.map(cnn_maxpool_size, function(x)
    -- Each element in the maxpool sizes list must be a pair of integers
    local t = table.map(string.split(x, '[^,]+'), tonumber)
    table.extend_with_last_element(t, 2)
    return t
end)
cnn_batch_norm = table.map(cnn_batch_norm, function(x) return x == 'true'end)

-- Initialize random seeds
math.randomseed(opt.seed)
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

-- Auxiliar function that creates convolutional block
function convBlock(depth_in, depth_out,  -- Input & output channels/filters
		   kernel_w, kernel_h,   -- Size of the convolution kernels
		   pool_w, pool_h,       -- Size of the pooling windows
		   activation, batch_norm, dropout)
   activation = activation or 'relu'
   batch_norm = batch_norm or false
   dropout = dropout or 0
   local block = nn.Sequential()
   -- Spatial dropout to the input of the convolutional block
   if dropout > 0 then
      block:add(nn.SpatialDropout(dropout))
   end
   -- Spatial 2D convolution. Image is padded with zeroes so that the output
   -- has the same size as the input / stride.
   block:add(cudnn.SpatialConvolution(
		depth_in, depth_out,
		kernel_w, kernel_h,
		1, 1,
		(kernel_w - 1) / 2, (kernel_h - 1) / 2))
   -- Batch normalization
   if batch_norm then
      block:add(cudnn.SpatialBatchNormalization(depth_out))
   end
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
   else
      assert(false, string.format('Unknown activation function %s', activation))
   end
   -- Max pooling
   if pool_w > 0 and pool_h > 0 then
      block:add(nn.SpatialMaxPooling(pool_w, pool_h, pool_w, pool_h))
   end
   return block
end

function computeSizeAfterPooling(input_size, pool_size)
   if pool_size < 2 then
      return input_size
   else
      return math.floor((input_size - pool_size) / pool_size + 1)
   end
end

local model = nn.Sequential()
-- First convolutional layer
model:add(convBlock(opt.input_channels, cnn_num_features[1],
		    cnn_kernel_size[1][1], cnn_kernel_size[1][2],
		    cnn_maxpool_size[1][1], cnn_maxpool_size[1][2],
		    cnn_type[1], cnn_batch_norm[1], cnn_dropout[1]))
-- Used to compute the height of the images after all the convolutions
local curr_h = computeSizeAfterPooling(opt.input_height, cnn_maxpool_size[1][2])
-- Append remaining convolutional layer blocks
for i=2,cnn_layers do
   model:add(convBlock(cnn_num_features[i-1], cnn_num_features[i],
		       cnn_kernel_size[i][1], cnn_kernel_size[i][2],
		       cnn_maxpool_size[i][1], cnn_maxpool_size[i][2],
		       cnn_type[i], cnn_batch_norm[i], cnn_dropout[i]))
   curr_h = computeSizeAfterPooling(curr_h, cnn_maxpool_size[i][2])
end
-- Append recurrent layers
local rnn_input_dim = cnn_num_features[cnn_layers - 1] * curr_h
model:add(nn.Transpose({2,4},{1,2}))
model:add(nn.Contiguous())
model:add(nn.Reshape(-1, rnn_input_dim, true))
if opt.rnn_type == 'blstm' then
   model:add(cudnn.BLSTM(rnn_input_dim, opt.rnn_units, opt.rnn_layers, false,
			 opt.rnn_dropout))
else
   model:add(cudnn.BGRU(rnn_input_dim, opt.rnn_units, opt.rnn_layers, false,
			opt.rnn_dropout))
end
model:add(nn.Contiguous())
model:add(nn.Reshape(-1, opt.rnn_units * 2, false))
if opt.linear_dropout > 0 then
   model:add(nn.Dropout(linear_dropout))
end
model:add(nn.Linear(opt.rnn_units * 2, opt.output_size))

-- Save model to disk
torch.save(opt.output_file, model)
