local SpatialMaxPooling, parent = torch.class('laia.cudnn.SpatialMaxPoolingHWNC', 'laia.cudnn._PoolingHWNC')

function SpatialMaxPooling:updateOutput(input)
   self.mode = 'CUDNN_POOLING_MAX'
   return parent.updateOutput(self, input)
end

function SpatialMaxPooling:__tostring__()
   return nn.SpatialMaxPooling.__tostring__(self)
end
