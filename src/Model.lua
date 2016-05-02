require 'nn'
require 'rnn'

function createModel(sample_height, num_labels)
   local ks = {3, 3, 3, 3, 3, 3, 2}
   local ss = {1, 1, 1, 1, 1, 1, 1}
   --local nm = {64, 128, 256, 256, 512, 512, 512}
   local nm = {16, 16, 32, 32, 64, 128, 128}
   local nh = {256, 256}

   function convBlock(depth_in, depth_out, size, stride, batch_norm)
      batch_norm = batch_norm or false
      local block = nn.Sequential()
      -- Spatial 2D convolution, Image is padded with zeroes so that the output
      -- has the same size as the input / stride
      block:add(nn.SpatialConvolution(depth_in, depth_out, size, size,
				      stride, stride,
				      (size - 1) / 2, (size - 1) / 2))
      -- Batch normalization (optional)
      if batch_norm then
	 block:add(nn.SpatialBatchNormalization(depth_out))
      end
      -- Parametric Rectifier Linear Unit
      block:add(nn.ReLU(true))
      return block
   end

   local model = nn.Sequential()

   model:add(convBlock(1, nm[1], ks[1], ss[1]))
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(convBlock(nm[1], nm[2], ks[2], ss[2]))
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   model:add(convBlock(nm[2], nm[3], ks[3], ss[3], true))
   model:add(convBlock(nm[3], nm[4], ks[4], ss[4]))
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   --model:add(convBlock(nm[4], nm[5], ks[5], ss[5], true))
   --model:add(convBlock(nm[5], nm[6], ks[6], ss[6]))

   model:add(nn.SplitTable(4))
   model:add(nn.Sequencer(nn.Reshape(-1, true)))
   model:add(nn.Sequencer(nn.Dropout(0.5)))
   model:add(nn.BiSequencer(nn.LSTM(nm[4] * sample_height / 8, nh[1]),
			    nn.LSTM(nm[4] * sample_height / 8, nh[1]),
			    nn.CAddTable()))
   model:add(nn.Sequencer(nn.Dropout(0.5)))
   model:add(nn.BiSequencer(nn.LSTM(nh[1], nh[2]),
			    nn.LSTM(nh[1], nh[2]),
			    nn.CAddTable()))
   model:add(nn.Sequencer(nn.Dropout(0.5)))
   model:add(nn.Sequencer(nn.Linear(nh[2], num_labels + 1)))
   model:add(nn.JoinTable(1))
   return model
end

-- function createModel2(sample_height, num_labels)
--    local ks = {3, 3, 3, 3, 3, 3, 2}
--    local ss = {1, 1, 1, 1, 1, 1, 1}
--    local nm = {16, 16, 32, 32, 64, 128, 128}
--    local nh = {256, 256}

--    function convBlock(depth_in, depth_out, size, stride, batch_norm)
--       batch_norm = batch_norm or false
--       local block = nn.Sequential()
--       -- Spatial 2D convolution, Image is padded with zeroes so that the output
--       -- has the same size as the input / stride
--       block:add(nn.SpatialConvolution(depth_in, depth_out, size, size,
-- 				      stride, stride,
-- 				      (size - 1) / 2, (size - 1) / 2))
--       -- Batch normalization (optional)
--       if batch_norm then
-- 	 block:add(nn.SpatialBatchNormalization(depth_out))
--       end
--       -- Parametric Rectifier Linear Unit
--       block:add(nn.ReLU(true))
--       return block
--    end

--    local model = nn.Sequential()

--    model:add(convBlock(1, nm[1], ks[1], ss[1]))
--    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
--    model:add(convBlock(nm[1], nm[2], ks[2], ss[2]))
--    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
--    model:add(convBlock(nm[2], nm[3], ks[3], ss[3], true))
--    model:add(convBlock(nm[3], nm[4], ks[4], ss[4]))
--    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

--    --model:add(convBlock(nm[4], nm[5], ks[5], ss[5], true))
--    --model:add(convBlock(nm[5], nm[6], ks[6], ss[6]))

--    model:add(nn.SplitTable(4))
--    model:add(nn.Sequencer(nn.Reshape(-1, true)))
--    model:add(nn.Sequencer(nn.Dropout(0.5))
--    model:add(nn.BiSequencer(nn.LSTM(nm[4] * sample_height / 8, nh[1]),
-- 			    nn.LSTM(nm[4] * sample_height / 8, nh[1]),
-- 			    nn.CAddTable()))
--    model:add(nn.Sequencer(nn.Dropout(0.5))
--    model:add(nn.BiSequencer(nn.LSTM(nh[1], nh[2]),
-- 			    nn.LSTM(nh[1], nh[2]),
-- 			    nn.CAddTable()))
--    model:add(nn.Sequencer(nn.Dropout(0.5))
--    model:add(nn.Sequencer(nn.Linear(nh[2], num_labels + 1)))
--    model:add(nn.JoinTable(1))
--    return model
-- end

-- function createModel_BLSTM(sample_height, num_labels, num_layers, num_hidden,
-- 			   batch_norm, dropout)
--    batch_norm = batch_norm or false
--    dropout = dropout or 0.0

--    local model = nn.Sequential()
--    model:add(nn.SplitTable(4))
--    model:add(nn.Sequencer(nn.Reshape(-1, true)))
--    model:add(nn.BiSequencer(nn.LSTM(sample_height, num_hidden),
-- 			    nn.LSTM(sample_height, num_hidden),
-- 			    nn.CAddTable()))
--    for i=2,num_hidden do
--       if batch_norm then model:add(nn.BatchNormalization(num_hidden)) end
--       if dropout > 0.0 then model:add(nn.Sequencer(nn.Dropout(dropout))) end
--       model:add(nn.BiSequencer(nn.LSTM(num_hidden, num_hidden),
-- 			       nn.LSTM(num_hidden, num_hidden),
-- 			       nn.CAddTable()))
--    end
--    model:add(nn.Sequencer(nn.Linear(num_hidden, num_labels + 1)))
--    model:add(nn.JoinTable(1))
--    return model
-- end
