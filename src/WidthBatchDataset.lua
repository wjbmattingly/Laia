require 'torch';
require 'hdf5';

local WidthBatchDataset = torch.class('WidthBatchDataset');

function WidthBatchDataset:__init(hdf5_path, centered_patch)
   self._hdf5_path = hdf5_path;
   self._hf = hdf5.open(self._hdf5_path, 'r');
   self._centered_patch = centered_patch or false
   assert(self._hf ~= nil, string.format('Dataset %q cannot be read!',
					 self._hdf5_path));
   -- Load sample IDs and weights
   self._num_samples = 0;
   local samples_aux = {};
   for k, _ in next,self._hf:read('/')._children do
      local img = self._hf:read(string.format('/%s/img', k)):all();
      table.insert(samples_aux, {k, img:size()[3]});
      self._num_samples = self._num_samples + 1;
   end;
   table.sort(samples_aux, function(a, b) return a[2] < b[2]; end);
   self._samples = {};
   for i=1,self._num_samples do
      table.insert(self._samples, samples_aux[i][1]);
   end;
   self._idx = 0;
end;

function WidthBatchDataset:numSamples()
   return self._num_samples;
end;

function WidthBatchDataset:nextBatch(batch_size)
   batch_size = batch_size or 1;
   assert(self._num_samples > 0, string.format('The dataset is empty!'));
   assert(batch_size > 0, string.format('Batch size must be greater than 0'));
   -- Get sizes of each sample in the batch
   local n_channels = nil;
   local batch_sizes = torch.LongTensor(batch_size, 3);
   for i=0,(batch_size-1) do
      local j = 1 + (i + self._idx) % self._num_samples;
      -- Read ground truth and images
      local gt = self._hf:read(
	 string.format('/%s/gt', self._samples[j])):all();
      local img = self._hf:read(
	 string.format('/%s/img', self._samples[j])):all();
      -- Check the number of channels is correct
      n_channels = (n_channels or img:size()[1]);
      assert(n_channels == img:size()[1], string.format(
		'Wrong number of channels in sample %q (got %d, expected %d)',
		k, img:size()[1], n_channels));
      batch_sizes[{i+1,1}] = gt:size()[1];      -- GT length
      batch_sizes[{i+1,2}] = img:size()[2];     -- Image height
      batch_sizes[{i+1,3}] = img:size()[3];     -- Image width
   end;
   -- Copy data into single batch tensors
   local max_sizes = torch.max(batch_sizes, 1);
   local batch_img = torch.Tensor(batch_size, n_channels, max_sizes[{1,2}],
				  max_sizes[{1,3}]):zero();
   local batch_gt  = torch.Tensor(batch_size, max_sizes[{1,1}]):zero();
   for i=0,(batch_size-1) do
      local j = 1 + (i + self._idx) % self._num_samples;
      local gt = self._hf:read(
	 string.format('/%s/gt', self._samples[j])):all();
      local img = self._hf:read(
	 string.format('/%s/img', self._samples[j])):all();
      local dy = 0; local dx = 0;
      if self._centered_patch then
	 dy = math.floor((max_sizes[{1,2}] - img:size()[2]) / 2);
	 dx = math.floor((max_sizes[{1,3}] - img:size()[3]) / 2);
      end;
      batch_img[i+1]:sub(1, n_channels,
			 dy + 1, dy + img:size()[2],
			 dx + 1, dx + img:size()[3]):copy(img);
      batch_gt[i+1]:sub(1, gt:size()[1]):copy(gt);
   end;
   -- Increase index for next batch
   self._idx = (self._idx + batch_size) % self._num_samples;
   collectgarbage();
   return batch_img, batch_gt, batch_sizes;
end;
