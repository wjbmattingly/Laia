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
require 'xlua';

require 'WidthBatcher';
require 'CurriculumBatcher';
require 'RandomBatcher';
require 'utils';

local seed = 1234;
local use_gpu = true;
local use_cudnn = true;
local BATCH_SIZE = 16;
local SAMPLE_HEIGHT = 64;
local NUM_CHARS = 78;
local grad_clip = 3;
local rmsprop_opts = {
   learningRate = 0.001,
   alpha = 0.95
}
local learning_rate_decay = 0.97;
local learning_rate_decay_after = 10;
local curriculum_lambda_start = 3;
local curriculum_lambda_iters = 10;

torch.manualSeed(seed);
math.randomseed(seed);

function createModel(sample_height, num_labels)
   local ks = {3, 3, 3, 3, 3, 3, 2}
   local ss = {1, 1, 1, 1, 1, 1, 1}
   --local nm = {64, 128, 256, 256, 512, 512, 512}
   local nm = {16, 16, 32, 32, 64, 128, 128}
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

   model:add(convBlock(1, nm[1], ks[1], ss[1]));
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2));
   model:add(convBlock(nm[1], nm[2], ks[2], ss[2]));
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2));
   model:add(convBlock(nm[2], nm[3], ks[3], ss[3], true));
   model:add(convBlock(nm[3], nm[4], ks[4], ss[4]));
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2));

   --model:add(convBlock(nm[4], nm[5], ks[5], ss[5], true));
   --model:add(convBlock(nm[5], nm[6], ks[6], ss[6]));

   model:add(nn.SplitTable(4));
   model:add(nn.Sequencer(nn.Reshape(-1, true)));
   model:add(nn.BiSequencer(nn.LSTM(nm[4] * sample_height / 8, nh[1]),
			    nn.LSTM(nm[4] * sample_height / 8, nh[1]),
			    nn.CAddTable()));
   model:add(nn.BiSequencer(nn.LSTM(nh[1], nh[2]),
			    nn.LSTM(nh[1], nh[2]),
			    nn.CAddTable()));
   model:add(nn.Sequencer(nn.Linear(nh[2], num_labels + 1)));
   model:add(nn.JoinTable(1));
   return model;
end;




local model = createModel(SAMPLE_HEIGHT, NUM_CHARS);
if use_gpu then
   model = model:cuda();
   if use_cudnn then
      cudnn.convert(model, cudnn);
   end;
end;

parameters, gradParameters = model:getParameters();

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


-- local dt = WidthBatcher('../data/iam/train.h5', true);

-- local dt = RandomBatcher('../data/iam/train.h5', true);
-- dt:shuffle();

-- local dt = CurriculumBatcher('../data/iam/train.h5', true);
-- dt:sample(3, 5);

local dt = CurriculumBatcher('../data/iam/train.h5', true);
local dv = RandomBatcher('../data/iam/valid.h5', true);

for epoch=1,1000 do
   -- Apply learning rate decay
   if epoch > learning_rate_decay_after then
      rmsprop_opts.learningRate =
	 rmsprop_opts.learningRate * learning_rate_decay;
   end;
   -- Curriculum learning
   if epoch <= curriculum_lambda_iters then
      dt:sample(curriculum_lambda_start *
		   (1.0 - (epoch - 1) / curriculum_lambda_iters), 5);
   else
      dt:sample(0, 5);
   end;

   local train_loss_epoch = 0.0;
   for batch=1,dt:numSamples(),BATCH_SIZE do
      local batch_img, batch_gt, batch_sizes = dt:next(BATCH_SIZE);
      if use_gpu then
	 batch_img = batch_img:cuda();
      end;

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
	    loss = table.reduce(gpu_ctc(output, grad_output, batch_gt, sizes),
				operator.add, 0)
	 else
	    output = output:float()
	    grad_output = grad_output:float()
	    loss = table.reduce(cpu_ctc(output, grad_output, batch_gt, sizes),
				operator.add, 0)
	 end;
	 -- Make loss function (and output gradients) independent of batch size
	 -- and sequence length.
	 loss = loss / (BATCH_SIZE * seq_len)
	 grad_output:div(BATCH_SIZE * seq_len)
	 -- Compute gradients of the loss function respect the parameters
	 model:backward(batch_img, grad_output)
	 -- Clip gradients
	 if grad_clip > 0 then
	    gradParameters:clamp(-grad_clip, grad_clip)
	 end
	 train_loss_epoch = train_loss_epoch + BATCH_SIZE * loss;
	 return loss, gradParameters
      end;
      optim.rmsprop(feval, parameters, rmsprop_opts)
      xlua.progress(batch + BATCH_SIZE - 1, dt:numSamples());
   end;
   local lastGradParameters = gradParameters:clone();

   -- VALIDATION
   dv:shuffle();
   local valid_loss_epoch = 0.0;
   for batch=1,dv:numSamples(),BATCH_SIZE do
      local batch_img, batch_gt, batch_sizes = dv:next(BATCH_SIZE);
      if use_gpu then
	 batch_img = batch_img:cuda();
      end;

      local output = model:forward(batch_img);
      local sizes = {}
      local seq_len = output:size()[1] / BATCH_SIZE;
      for i=1,BATCH_SIZE do table.insert(sizes, seq_len) end;

      local grad_output = output:clone():zero();
      local loss = 0;
      -- Compute loss function and gradients respect the output
      if use_gpu then
	 loss = table.reduce(gpu_ctc(output, grad_output, batch_gt, sizes),
			     operator.add, 0)
      else
	 output = output:float()
	 grad_output = grad_output:float()
	 loss = table.reduce(cpu_ctc(output, grad_output, batch_gt, sizes),
			     operator.add, 0);
      end;
      -- Make loss function (and output gradients) independent of batch size
      -- and sequence length.
      loss = loss / (BATCH_SIZE * seq_len);
      valid_loss_epoch = valid_loss_epoch + BATCH_SIZE * loss;
      xlua.progress(batch + BATCH_SIZE - 1, dv:numSamples());
   end;

   train_loss_epoch = train_loss_epoch / dt:numSamples()
   valid_loss_epoch = valid_loss_epoch / dv:numSamples()
   --local gmin, gmax, mass = torch.sumarizeMagnitudes(
   --gradParameters, 0.85, 100);
   --print(string.format('Epoch = %-5d  Avg. Train Loss = %7.4f  -- ' ..
   -- '%5.2f%% of gradients are in range (%g, %g]',
   --epoch, train_loss_epoch, mass * 100, gmin, gmax))
   print(string.format('Epoch = %-5d  Avg. Train Loss = %7.4f  ' ..
			  'Avg. Valid Loss = %7.4f',
		       epoch, train_loss_epoch, valid_loss_epoch))
   -- Save model
   torch.save(string.format('model_epoch%05d.net', epoch), model)
end
