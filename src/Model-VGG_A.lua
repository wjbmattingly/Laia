require 'cudnn'
require 'src.BiLSTM'
require 'src.BGRU'

function createModel(sample_height, num_labels)
  local ks = {3, 3, 3, 3, 3, 3, 3, 3}
  local ss = {1, 1, 1, 1, 1, 1, 1, 1}
  local nm = {64, 128, 256, 256, 512, 512, 512, 512}
  local nh = {256}

  function convBlock(depth_in, depth_out, size, stride, batch_norm)
    batch_norm = batch_norm or false
    local block = nn.Sequential()
    
    -- Spatial 2D convolution. Image is padded with zeroes so that the output
    -- has the same size as the input / stride
    block:add(nn.SpatialConvolution(depth_in, depth_out, 
                                    size, size,
                                    stride, stride,
                                    (size - 1) / 2, (size - 1) / 2))
    
    -- Batch normalization
    if batch_norm then
      block:add(nn.SpatialBatchNormalization(depth_out))
    end
    
    -- Parametric Rectifier Linear Unit
    block:add(nn.LeakyReLU(true))
    
    return block
  end

  local model = nn.Sequential()

  -- CNN part
  model:add(convBlock(1, nm[1], ks[1], ss[1]), true)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --
  model:add(convBlock(nm[1], nm[2], ks[2], ss[2]), true)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --
  model:add(convBlock(nm[2], nm[3], ks[3], ss[3]), true)
  model:add(convBlock(nm[3], nm[4], ks[4], ss[4]), true)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --
  model:add(convBlock(nm[4], nm[5], ks[5], ss[5]), true)
  model:add(convBlock(nm[5], nm[6], ks[6], ss[6]), true)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --
  model:add(convBlock(nm[6], nm[7], ks[7], ss[7]), true)
  model:add(convBlock(nm[7], nm[8], ks[8], ss[8]), true)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))       -- 512 x 2
  
  -- RNN part
  model:add(nn.Transpose({2,4},{1,2}))
  model:add(nn.Contiguous())
  model:add(nn.Reshape(-1, 1024,true))
  model:add(nn.Dropout(0.5))
  model:add(cudnn.BiLSTM(1024, nh[1], 3, nil, 0.5))
  --model:add(cudnn.BGRU(nh[1], nh[1], 3, nil, 0.5))
  model:add(nn.Dropout(0.5))
  model:add(nn.Contiguous())  
  model:add(nn.Reshape(-1, nh[1]*2, false))
  model:add(nn.Linear(nh[1]*2, num_labels + 1))

  return model
end