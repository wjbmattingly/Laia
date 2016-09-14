require 'src.CachedBatcher'

local RandomBatcher, Parent = torch.class('RandomBatcher', 'CachedBatcher')

function RandomBatcher:__init(img_list, cfg)
  Parent.__init(self, img_list, cfg)
end

--[[
   Use this method to shuffle the samples to serve following batches in a
   different order. This should be called at the beginning of each EPOCH to
   process the whole dataset in a different order.
--]]
function RandomBatcher:shuffle()
  for i=1,(self._num_samples-1) do
    -- DO NOT USE math.random(self._num_samples - i), rely only on torch
    -- random generators, instead. The state of torch RNG can be stored and
    -- recovered for greater replicability.
    --local j = math.random(self._num_samples - i)
    local j = 1 + math.floor(torch.uniform() * (self._num_samples - i))
    self._samples[i], self._samples[j] = self._samples[j], self._samples[i]
    self._imglist[i], self._imglist[j] = self._imglist[j], self._imglist[i]
  end
  self._idx = 0
  self:clearCache()
end

function RandomBatcher:epochReset()
  self:shuffle()
end

return RandomBatcher
