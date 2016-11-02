local CurriculumBatcher, Parent = torch.class('laia.CurriculumBatcher',
					      'laia.CachedBatcher')

function CurriculumBatcher:__init(opt)
  Parent.__init(self, opt)
end

--[[
-- THESE ARE NOT OPTIONS OF THE BATCHER OBJECT, BUT THE TRAINING STRATEGY, SO
-- MOVE THESE OPTIONS TO THE TRAINER.
function CurriculumBatcher:registerOptions(parser)
  parser:option(
    '--curriculum_lambda',
    'This parameter controls the smoothness of the distribution used to ' ..
      'select samples according to their length. If lambda = 0, all samples ' ..
      'have the same likelihood. When lambda increases, shorter samples are ' ..
      'more likely to be selected. More precisely, the likelihood of a ' ..
      'sample is proportional to max(m, length)^(-lambda) ' ..
      '(see --curriculum_min_length).',
    0.0, tonumber)
    :argname('<lambda>')
    :overwrite(false)
    :ge(0.0)
  parser:option(
    '--curriculum_min_length',
    'This parameter avoids that very short sequences have an excessively ' ..
      'high likelihood to be selected.',
    1.0, tonumber)
    :argname('<m>')
    :overwrite(false)
    :ge(1.0)
end
--]]

function CurriculumBatcher:load(img_list, gt_file, symbols_table)
  assert(gt_file ~= nil and symbols_table ~= nil,
	 'CurriculumBatcher needs the ground-truth transcripts!')
  Parent.load(self, img_list, gt_file, symbols_table)
  -- This factor is just a denominator to make sure that exponentiations
  -- during sampling do not return infinity values.
  self._N = 1 + table.reduce(
    self._gt, function(acc, x) return math.max(acc, #x) end, 0)
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
  lambda = tonumber(lambda) or 0
  m = tonumber(m) or 1
  assert(lambda >= 0, 'Parameter lambda must be greater than or equal to 0')
  assert(m >= 1, 'Parameter m must be greater than or equal to 1')
  -- Compute likelihood of choosing each of the samples.
  local likelihoods = {}
  for k, gt in pairs(self._gt) do
    likelihoods[k] = math.pow(math.max(m, #gt) / self._N, -lambda)
  end
  local tot_likelihood = table.reduce(likelihoods, operator.add, 0.0)
  -- Select _num_samples samples according to their likelihood.
  self._samples = {}
  for n=1,self._num_samples do
    table.insert(self._samples,
		 table.weighted_choice(likelihoods, tot_likelihood))
    local k = self._samples[#self._samples]
  end
  self._idx = 0
  self:clearCache()
end

function CurriculumBatcher:epochReset(epoch_opt)
  epoch_opt = epoch_opt or {}
  assert(type(epoch_opt) == 'table', 'epochReset options must be a table!')
  self:sample(epoch_opt.lambda, epoch_opt.m)
end

return CurriculumBatcher
