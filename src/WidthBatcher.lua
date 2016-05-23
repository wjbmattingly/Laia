require 'src.CachedBatcher'

local WidthBatcher, Parent = torch.class('WidthBatcher', 'CachedBatcher')

function WidthBatcher:__init(hdf5_path, centered_patch, cache_max_size,
			     cache_gpu)
   Parent.__init(self, hdf5_path, centered_patch, cache_max_size, cache_gpu)
   -- Load sample IDs and width
   local sample_width = {}
   for k, _ in next,self._hf:read('/')._children do
      local img = self._hf:read(string.format('/%s/img', k)):all()
      table.insert(sample_width, {k, img:size()[3]})
   end
   -- Sort samples by increasing width
   table.sort(sample_width, function(a, b) return a[2] > b[2] end)
   self._samples = {}
   for i=1,self._num_samples do
      table.insert(self._samples, sample_width[i][1])
   end
end
