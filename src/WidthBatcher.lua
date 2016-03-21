require 'cutorch';
require 'hdf5';
require 'torch';

local WidthBatcher = torch.class('WidthBatcher');

function WidthBatcher:__init(hdf5_path, centered_patch, cache_max_size,
			     cache_gpu)
   self._hdf5_path = hdf5_path;
   self._centered_patch = centered_patch or false
   self._cache_max_size = cache_max_size or 512
   self._cache_gpu = cache_gpu or -1
   self._hf = hdf5.open(self._hdf5_path, 'r');
   assert(self._hf ~= nil,
	  string.format('Dataset %q cannot be read!', self._hdf5_path));
   -- Load sample IDs and weights
   self._num_samples = 0;
   local samples_aux = {};
   for k, _ in next,self._hf:read('/')._children do
      local img = self._hf:read(string.format('/%s/img', k)):all();
      table.insert(samples_aux, {k, img:size()[3]});
      self._num_samples = self._num_samples + 1;
   end;
   -- Sort samples by increasing width
   table.sort(samples_aux, function(a, b) return a[2] < b[2]; end);
   self._samples = {};
   for i=1,self._num_samples do
      table.insert(self._samples, samples_aux[i][1]);
   end;
   self._idx = 0;
   self._cache_img = {}
   self._cache_gt = {}
   self._cache_idx = 0;
   self._cache_size = 0;
end;

function WidthBatcher:numSamples()
   return self._num_samples;
end;

function WidthBatcher:_global2cacheIndex(idx)
   if #self._cache_img < 1 then return -1 end
   if idx < 1 or idx > self._num_samples then return -1; end
   idx = idx - 1
   local cacheEnd = (self._cache_idx + #self._cache_img) % self._num_samples
   if self._cache_idx < cacheEnd then
      -- Cache is contiguous: [i, i+1, ..., i + M]
      if idx >= self._cache_idx and idx < cacheEnd then
	 return 1 + idx - self._cache_idx
      else
	 return -1
      end
   else
      -- The cache is circular: [i, i + 1, ..., i + N, 0, ..., M - N - 1]
      if idx >= self._cache_idx or idx < cacheEnd then
	 return 1 + (idx - self._cache_idx) % self._num_samples
      else
	 return -1
      end
   end
end;

function WidthBatcher:_cache2globalIndex(idx)
   if idx < 1 or idx > #self._cache_img then return -1; end
   return 1 + (self._cache_idx + idx - 1) % self._num_samples
end;

function WidthBatcher:_fillCache(idx)
   -- If the requested image is not available in the cache, fill the cache
   -- with it and all the following images until the cache size increases up to
   -- self._cache_max_size MB (approx)
   if self:_global2cacheIndex(idx) < 0 then
      idx = idx - 1
      self._cache_img = {}
      self._cache_gt = {}
      self._cache_size = 0
      self._cache_idx = idx  -- First cached image = first in the batch
      -- Allocate memory in the GPU device self._cache_gpu, the current device
      -- is saved and restored later.
      local old_gpu = -1
      if self._cache_gpu >= 0 then
	 old_gpu = cutorch.getDevice();
	 cutorch.setDevice(self._cache_gpu)
      end
      while self._cache_size < self._cache_max_size and
      #self._cache_img < self._num_samples do
	 local j = 1 + (#self._cache_img + idx) % self._num_samples
	 -- Add image and ground truth to the cache
	 table.insert(self._cache_img, self._hf:read(
	     string.format('/%s/img', self._samples[j])):all():float())
	 table.insert(self._cache_gt, torch.totable(
	     self._hf:read(string.format('/%s/gt', self._samples[j])):all()))

	 -- Store data in the GPU, if requested.
	 if self._cache_gpu >= 0 then
	    self._cache_img[#self._cache_img] =
	       self._cache_img[#self._cache_img]:cuda()
	 end;
	 -- Increase size of the cache (in MB)
	 self._cache_size = self._cache_size +
	    self._cache_img[#self._cache_img]:storage():size() * 4 / 1048576 +
	    #(self._cache_gt[#self._cache_gt]) * 8 / 1048576;
      end;
      -- Restore value of the current GPU device
      if old_gpu >= 0 then
	 cutorch.setDevice(old_gpu)
      end
   end;
end;

function WidthBatcher:next(batch_size)
   batch_size = batch_size or 1;
   assert(batch_size > 0, string.format('Batch size must be greater than 0'));
   assert(self._num_samples > 0, string.format('The dataset is empty!'));
   -- Get sizes of each sample in the batch
   local n_channels = nil;
   local batch_sizes = torch.LongTensor(batch_size, 2);
   for i=0,(batch_size-1) do
      local j = 1 + (i + self._idx) % self._num_samples;
      self:_fillCache(j);
      local cacheIdx = self:_global2cacheIndex(j);
      local img = self._cache_img[cacheIdx];
      -- Check the number of channels is correct
      n_channels = (n_channels or img:size()[1]);
      assert(n_channels == img:size()[1], string.format(
		'Wrong number of channels in sample %q (got %d, expected %d)',
		k, img:size()[1], n_channels));
      batch_sizes[{i+1,1}] = img:size()[2];     -- Image height
      batch_sizes[{i+1,2}] = img:size()[3];     -- Image width
   end;
   -- Copy data into single batch tensors
   local max_sizes = torch.max(batch_sizes, 1);
   local batch_img = torch.Tensor(batch_size, n_channels, max_sizes[{1,1}],
				  max_sizes[{1,2}]):zero();
   local batch_gt  = {};
   local old_gpu = -1
   if self._cache_gpu >= 0 then
      old_gpu = cutorch.getDevice()
      cutorch.setDevice(self._cache_gpu)
      batch_img = batch_img:cuda()
   end
   for i=0,(batch_size-1) do
      local j = 1 + (i + self._idx) % self._num_samples;
      self:_fillCache(j);
      local img = self._cache_img[self:_global2cacheIndex(j)];
      local gt  = self._cache_gt[self:_global2cacheIndex(j)];
      local dy = 0; local dx = 0;
      if self._centered_patch then
	 dy = math.floor((max_sizes[{1,1}] - img:size()[2]) / 2);
	 dx = math.floor((max_sizes[{1,2}] - img:size()[3]) / 2);
      end;
      batch_img[i+1]:sub(1, n_channels,
			 dy + 1, dy + img:size()[2],
			 dx + 1, dx + img:size()[3]):copy(img);
      table.insert(batch_gt, gt);
   end;
   if old_gpu >= 0 then
      cutorch.setDevice(old_gpu)
   end
   -- Increase index for next batch
   self._idx = (self._idx + batch_size) % self._num_samples;
   collectgarbage();
   return batch_img, batch_gt, batch_sizes;
end;
