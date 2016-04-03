require 'CachedBatcher';

local RandomBatcher, Parent = torch.class('RandomBatcher', 'CachedBatcher');

function RandomBatcher:__init(hdf5_path, centered_patch, cache_max_size,
			     cache_gpu)
   Parent.__init(self, hdf5_path, centered_patch, cache_max_size, cache_gpu)
end;

--[[
   Use this method to shuffle the samples to serve following batches in a
   different order. This should be called at the beginning of each EPOCH to
   process the whole dataset in a different order.
--]]
function RandomBatcher:shuffle()
   for i=1,(self._num_samples-1) do
      local j = math.random(self._num_samples - i);
      self._samples[i], self._samples[j] = self._samples[j], self._samples[i];
   end;
   self._idx = 0;
   self:clearCache();
end;
