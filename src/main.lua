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

function convBlock(depth_in, depth_out, size, stride, batch_norm)
   batch_norm = batch_norm or false;
   local block = nn.Sequential();
   -- Spatial 2D convolution
   block:add(nn.SpatialConvolution(depth_in, depth_out, size, size,
					 stride, stride));
   -- Batch normalization (optional)
   if batch_norm then
      block:add(nn.SpatialBatchNormalization(depth_out));
   end;
   -- Parametric Rectifier Linear Unit
   block:add(nn.ReLU(true));
   return block;
end;


local ds = WidthBatchDataset('../data/iam/train.h5', true);
local batch_img, batch_gt, batch_sizes = ds:nextBatch(128);
batch_img = batch_img:float():cuda();
batch_gt = batch_gt:float():cuda();


local conv_model = nn.Sequential();
conv_model:add(convBlock(1, 64, 5, 1));
conv_model:add(convBlock(64, 128, 3, 1));
--conv_model:add(nn.ImageSequencer());

print( conv_model:forward(batch_image));
