require 'argparse';
require 'torch';
require 'nn';
require 'cutorch';
require 'cunn';
require 'cudnn';
require 'rnn';
require 'image';

require 'WidthBatchDataset';
require 'ImageSequencer';
require 'Permute';

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

local BATCH_SIZE = 64;
local SAMPLE_HEIGHT = 64;

local ds = WidthBatchDataset('../data/iam/train.h5', true);
local batch_img, batch_gt, batch_sizes = ds:nextBatch(BATCH_SIZE);
batch_img, batch_gt, batch_sizes = ds:nextBatch(BATCH_SIZE);
batch_img = batch_img:float():cuda();
batch_gt = batch_gt:float():cuda();


local model = nn.Sequential();
model:add(convBlock(1, 64, 5, 2));
model:add(convBlock(64, 128, 3, 2)); -- N x D x H x W
model:add(nn.SplitTable(4));         -- {N x D x H, N x D x H, ...}
model:add(nn.Sequencer(nn.Reshape(-1, true))); -- {N x (D*H), N x (D*H), ...}
model:add(nn.BiSequencer(nn.LSTM(SAMPLE_HEIGHT * 128 / 4, 500),
			 nn.LSTM(SAMPLE_HEIGHT * 128 / 4, 500),
			 nn.CAddTable()));
model:add(nn.BiSequencer(nn.LSTM(500, 250),
			 nn.LSTM(500, 250),
			 nn.CAddTable()));
model = model:cuda();

print(model:forward(batch_img));
