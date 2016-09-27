require 'laia.CachedBatcher'
require 'laia.utilities'

local CurriculumBatcher, Parent = torch.class('CurriculumBatcher',
					      'CachedBatcher')

function CurriculumBatcher:__init(hdf5_path, centered_patch, cache_max_size,
				  cache_gpu)
   Parent.__init(self, hdf5_path, centered_patch, cache_max_size, cache_gpu)
   -- Load GT length of each sample
   self._sample_length = {}
   for k, _ in next,self._hf:read('/')._children do
      local gt = self._hf:read(string.format('/%s/gt', k)):all()
      self._sample_length[k] = gt:size()[1]
   end
end

--[[
   Use this method to select _num_samples samples from the original dataset,
   according to a probability inversely proportional to the length their text
   transcription.

   Parameter ``lambda'' (default 0.0) controls the smoothness of the
   distribution, respect  the length of the samples. If lamba = 0, all samples
   have the same probability of being sampled, regardless of their length. If
   lambda increases, shorter samples are more likely to be sampled.

   Parameter ``m'' (default 1.0), is used to avoid that very short sequences
   have an excessively high likelihood to be selected.

   You should call this method at the beginning of each EPOCH, so that in each
   epoch you process _num_samples samples, with some repetitions most likely.

   See ``Curriculum Learning for Handwritten Text Line Recognition'', by
   Jeremoe Louradour and Christopher Kermovant.
--]]
function CurriculumBatcher:sample(lambda, m)
   lambda = lambda or 0
   m = m or 1
   assert(lambda >= 0, 'Parameter lambda must be greater than or equal to 0')
   assert(m >= 1, 'Parameter m must be greater than or equal to 1')
   -- Compute likelihood of choosing each of the samples.
   local likelihoods = {}
   for k, l in pairs(self._sample_length) do
      likelihoods[k] = 1.0 / math.pow(math.max(m, l), lambda)
   end
   local tot_likelihood = table.reduce(likelihoods, operator.add, 0.0)
   -- Select _num_samples samples according to their likelihood.
   self._samples = {}
   for n=1,self._num_samples do
      table.insert(self._samples,
                   table.weighted_choice(likelihoods, tot_likelihood))
   end
   self._idx = 0
   self:clearCache()
end
