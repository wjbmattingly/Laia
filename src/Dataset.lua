require 'torch';
require 'hdf5';

local Dataset = torch.class('Dataset');

function Dataset:__init(hdf5_path)
   self._hdf5_path = hdf5_path;
   self._hf = hdf5.open(self._hdf5_path, 'r');
   assert(self._hf ~= nil, string.format('Dataset %q cannot be read!',
					 self._hdf5_path));
   -- Load sample IDs
   self._samples = {};
   for k, _ in next,self._hf:read('/')._children do
      table.insert(self._samples, k);
   end;
   self._num_samples = table.getn(self._samples);
   self._idx = 1;
end;

function Dataset:numSamples()
   return self._num_samples;
end;

function Dataset:shuffle()
   for i=1,(self._num_samples-1) do
      local j = math.random(self._num_samples - i);
      self._samples[i], self._samples[j] = self._samples[j], self._samples[i]
   end;
   self._idx = 1;
end;

function Dataset:nextBatch(batch_size)
   assert(self._num_samples > 0, string.format('The dataset is empty!'))
   assert(batch_size > 0, string.format('Batch size must be greter than 0'))
   -- Get sizes of each sample in the batch
   local n_channels = nil;
   local batch_sizes = torch.LongTensor(batch_size, 3);
   for i=0,(batch_size-1) do
      local j = ((self._idx + i - 1) % self._num_samples) + 1;
      -- Read ground truth and images
      local gt = self._hf:read(
	 string.format('/%s/gt', self._samples[j])):all();
      local img = self._hf:read(
	 string.format('/%s/img', self._samples[j])):all();
      -- Check the number of channels is correct
      n_channels = (n_channels or img:size()[1]);
      assert(n_channels == img:size()[1], string.format(
		'Wrong number of channels in %q (got %d, expected %d)',
		self._samples[j], img:size()[1], n_channels));
      batch_sizes[{i+1,1}] = gt:size()[1];           -- GT length
      batch_sizes[{i+1,2}] = img:size()[2];          -- Image height
      batch_sizes[{i+1,3}] = img:size()[3];          -- Image width
   end;
   -- Copy data into single batch tensors
   local max_sizes = torch.max(batch_sizes, 1);
   local batch_img = torch.Tensor(batch_size, n_channels, max_sizes[{1,2}],
				  max_sizes[{1,3}]):zero();
   local batch_gt  = torch.Tensor(batch_size, max_sizes[{1,1}]):zero();
   for i=0,(batch_size-1) do
      local j = ((self._idx + i - 1) % self._num_samples) + 1;
      local gt = self._hf:read(
	 string.format('/%s/gt', self._samples[j])):all();
      local img = self._hf:read(
	 string.format('/%s/img', self._samples[j])):all();
      batch_img[i+1]:sub(1, n_channels, 1, batch_sizes[{i+1,2}],
			 1, batch_sizes[{i+1,3}]):copy(img);
      batch_gt[i+1]:sub(1, batch_sizes[{i+1,1}]):copy(gt);
   end;
   -- Increase index for next batch
   self._idx = ((self._idx + batch_size - 1) % self._num_samples) + 1;
   return batch_img, batch_gt, batch_sizes;
end;
