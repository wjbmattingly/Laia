require 'argparse';
require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';
require 'cudnn';
require 'rnn';
require 'image';
require 'warp_ctc';
require 'optim';

require 'WidthBatcher';

local use_gpu = false
local BATCH_SIZE = 4;
local SAMPLE_HEIGHT = 64;
local NUM_CHARS = 78;
local grad_clip = 0;

function createModel(sample_height, num_labels)
   local ks = {3, 3, 3, 3, 3, 3, 2}
   local ss = {1, 1, 1, 1, 1, 1, 1}
   local nm = {64, 128, 256, 256, 512, 512, 512}
   local nh = {256, 256}

   function convBlock(depth_in, depth_out, size, stride, batch_norm)
      batch_norm = batch_norm or false;
      local block = nn.Sequential();
      -- Spatial 2D convolution, Image is padded with zeroes so that the output
      -- has the same size as the input / stride
      block:add(nn.SpatialConvolution(depth_in, depth_out, size, size,
				      stride, stride,
				      (size - 1) / 2, (size - 1) / 2));
      -- Batch normalization (optional)
      if batch_norm then
	 block:add(nn.SpatialBatchNormalization(depth_out));
      end;
      -- Parametric Rectifier Linear Unit
      block:add(nn.ReLU(true));
      return block;
   end;

   local model = nn.Sequential();
   --[[
   model:add(convBlock(1, nm[1], ks[1], ss[1]));
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2));
   model:add(convBlock(nm[1], nm[2], ks[2], ss[2]));
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2));

   model:add(convBlock(nm[2], nm[3], ks[3], ss[3], true));
   model:add(convBlock(nm[3], nm[4], ks[4], ss[4]));
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2));

   model:add(convBlock(nm[4], nm[5], ks[5], ss[5], true));
   model:add(convBlock(nm[5], nm[6], ks[6], ss[6]));
   --]]
   model:add(nn.SplitTable(4));
   model:add(nn.Sequencer(nn.Reshape(-1, true)));
   model:add(nn.BiSequencer(nn.LSTM(sample_height, nh[1]),
			    nn.LSTM(sample_height, nh[1]),
			    nn.CAddTable()));
   --[[
   model:add(nn.BiSequencer(nn.LSTM(nm[6] * sample_height / 8, nh[1]),
			    nn.LSTM(nm[6] * sample_height / 8, nh[1]),
			    nn.CAddTable()));
   --]]
   model:add(nn.BiSequencer(nn.LSTM(nh[1], nh[2]),
			    nn.LSTM(nh[1], nh[2]),
			    nn.CAddTable()));
   model:add(nn.Sequencer(nn.Linear(nh[2], num_labels + 1)));
   -- model:add(nn.Sequencer(nn.SoftMax()));
   model:add(nn.JoinTable(1));
   return model;
end;



local ds = WidthBatcher('../data/iam/train.h5', true);

local model = createModel(SAMPLE_HEIGHT, NUM_CHARS);
if use_gpu then
   model = model:cuda();
end

parameters, gradParameters = model:getParameters()


local batch_img, batch_gt, batch_sizes = ds:next(BATCH_SIZE);



for epoch=1,1000 do
   if use_gpu then
      batch_img = batch_img:cuda();
   end

   local feval = function(x)
      assert (x == parameters)
      collectgarbage()
      gradParameters:zero()

      local output = model:forward(batch_img);
      local sizes = {}
      local seq_len = output:size()[1] / BATCH_SIZE;
      for i=1,BATCH_SIZE do table.insert(sizes, seq_len) end;

      local grad_output = output:clone():zero();
      local loss = 0;
      -- Compute loss function and gradients respect the output
      if use_gpu then
	 loss = table.reduce(gpu_ctc(output, grad_output, batch_gt, sizes), operator.add, 0)
      else
	 output = output:float()
	 grad_output = grad_output:float()
	 loss = table.reduce(cpu_ctc(output, grad_output, batch_gt, sizes), operator.add, 0)
      end
      -- Make loss function (and output gradients) independent of batch size and sequence length
      loss = loss / (BATCH_SIZE * seq_len)
      grad_output:div(BATCH_SIZE * seq_len)
      -- Compute gradients of the loss function respect the parameters
      model:backward(batch_img, grad_output)
      local gradParamAbs = torch.abs(gradParameters)

      print(loss)

      if grad_clip > 0 then
	 gradParameters:clip(-grad_clip, grad_clip)
      end
      return loss, gradParameters
   end
   optim.sgd(feval, parameters, {learningRate = 0.1})
end
