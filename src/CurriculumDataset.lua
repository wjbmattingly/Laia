require 'torch';
require 'hdf5';

local CurriculumDataset = torch.class('CurriculumDataset');

function CurriculumDataset:__init(hdf5_path)
   self._hdf5_path = hdf5_path;
   self._hf = hdf5.open(self._hdf5_path, 'r');
   assert(self._hf ~= nil, string.format('Dataset %q cannot be read!',
					 self._hdf5_path));
   -- Load sample IDs and weights
   self._samples = {};
   self._num_samples = 0;
   self._max_width = 0;
   for k, _ in next,self._hf:read('/')._children do
      local img = self._hf:read(string.format('/%s/img', k)):all();
      self._samples[k] = img:size()[3];  -- Sample Weight
      self._num_samples = self._num_samples + 1;
      if img:size()[3] > self._max_width then
	 self._max_width = img:size()[3];
      end;
   end;
end;

function CurriculumDataset:numSamples()
   return self._num_samples;
end;

function CurriculumDataset:_weightedChoice(likelihoods, tot_likelihood)
   local cut_likelihood = math.random() * tot_likelihood;
   local cum_likelihood = 0;
   for k, w in pairs(likelihoods) do
      if cum_likelihood + w >= cut_likelihood then
	 return k;
      end;
      cum_likelihood = cum_likelihood + w;
   end;
end;

function CurriculumDataset:nextBatch(batch_size, lambda)
   batch_size = batch_size or 1;
   lambda = lambda or 0;
   assert(self._num_samples > 0, string.format('The dataset is empty!'));
   assert(batch_size > 0, string.format('Batch size must be greater than 0'));
   assert(lambda >= 0,
	  string.format('Lambda must be greater than or equal to 0'));
   -- Select batch_size samples proportional to their sampling likelihood,
   -- which is proportional to 1 / (Width^Lambda). Samples will be selected
   -- without repetitions, unless the batch size is greater than the number
   -- of available samples.
   local samples = {};
   while #samples < batch_size do
      -- Compute likelihood of choosing each of the samples.
      local likelihoods = {};
      local tot_likelihood = 0.0;
      for k, w in pairs(self._samples) do
	 likelihoods[k] = math.pow(w / self._max_width, -lambda);
	 tot_likelihood = tot_likelihood + likelihoods[k];
	 print(w, likelihoods[k]);
      end;
      -- Select samples according to their likelihood.
      while tot_likelihood > 0 and #samples < batch_size do
	 local k = self:_weightedChoice(likelihoods, tot_likelihood);
	 tot_likelihood = tot_likelihood - likelihoods[k];
	 table.insert(samples, k);
	 likelihoods[k] = nil;
	 print(k);
      end;
   end;
   -- Get sizes of each sample in the batch
   local n_channels = nil;
   local batch_sizes = torch.LongTensor(batch_size, 3);
   for i=1,#samples do
      -- Read ground truth and images
      local gt = self._hf:read(string.format('/%s/gt', samples[i])):all();
      local img = self._hf:read(string.format('/%s/img', samples[i])):all();
      -- Check the number of channels is correct
      n_channels = (n_channels or img:size()[1]);
      assert(n_channels == img:size()[1], string.format(
		'Wrong number of channels in sample %q (got %d, expected %d)',
		k, img:size()[1], n_channels));
      batch_sizes[{i,1}] = gt:size()[1];           -- GT length
      batch_sizes[{i,2}] = img:size()[2];          -- Image height
      batch_sizes[{i,3}] = img:size()[3];          -- Image width
   end;
   -- Copy data into single batch tensors
   local max_sizes = torch.max(batch_sizes, 1);
   local batch_img = torch.Tensor(batch_size, n_channels, max_sizes[{1,2}],
				  max_sizes[{1,3}]):zero();
   local batch_gt  = torch.Tensor(batch_size, max_sizes[{1,1}]):zero();
   for i=1,#samples do
      local gt = self._hf:read(string.format('/%s/gt', samples[i])):all();
      local img = self._hf:read(string.format('/%s/img', samples[i])):all();
      batch_img[i]:sub(1, n_channels, 1, batch_sizes[{i,2}],
		       1, batch_sizes[{i,3}]):copy(img);
      batch_gt[i]:sub(1, batch_sizes[{i,1}]):copy(gt);
   end;
   -- Increase index for next batch
   collectgarbage();
   return batch_img, batch_gt, batch_sizes;
end;
