-- Statistical utility functions.
--
--------------------------------------------------------------------------------
-- BOOTSTRAPPING
--------------------------------------------------------------------------------
--
-- Example, how to compute CER with bootstrapped confidence intervals.
--
-- edit_ops = {1, 0, 2, 4, 2, 1, 4, 2, 3, 0, 3, 2, 1, 4}
-- ref_length = {12, 11, 9, 18, 12, 17, 10, 13, 12, 9, 11, 5, 1, 20}
--
-- -- Obtain empirical CER
-- cer = table.reduce(edit_ops, operator.add, 0) /
--       table.reduce(ref_length, operator.add, 0)
--
-- -- Obtain CER differences between each Bootstrap CER and the empirical CER.
-- -- CER differences are sorted in increasing order.
-- bootstrap_samples = laia.bootstrap_sample(#edit_ops, 1000)
-- bootstrap_edit_ops = laia.bootstrap_data(edit_ops, bootstrap_samples)
-- bootstrap_ref_length = laia.bootstrap_data(ref_length, bootstrap_samples)
-- cer_diffs = {}
-- for s=1:#bootstrap_samples do
--   local s_cer = table.reduce(bootstrap_edit_ops[s], operator.add, 0) /
--                 table.reduce(bootstrap_ref_length[s], operator.add, 0)
--   table.insert(cer_diffs, s_cer - cer)
-- end
-- table.sort(cer_diffs)
-- -- Compute bootstrap confidence intervals for the population CER.
-- -- Watch that lower/upper bounds are obtained from 97.5%/2.5%, respectively!
-- cer_lower = cer - cer_diffs[math.ceil(0.975 * #cer_diffs)]
-- cer_upper = cer - cer_diffs[math.ceil(0.025 * #cer_diffs)]
-- print(('The sample CER is %.2f, with a 95%% confidence interval in [%.2f, %.2f]'):format(cer, cer_lower, cer_upper))


-- Create `num_samples' boostrap samples from a set of `num_data' original
-- data items, using Case Resampling.
--
-- Each bootstrap sample has `num_data' elements randomly selected from the
-- range [1..num_data], i.e. resampling with repetitions.
--
-- See https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Case_resampling
local tds = require 'tds'

laia.bootstrap_sample = function(num_data, num_samples)
  local bootstrap_samples = tds.Vec()
  for s=1,num_samples do
	bootstrap_samples:insert(tds.Vec())
    for i=1,num_data do
	    bootstrap_samples[s]:insert(torch.random(num_data))
    end
  end
  return bootstrap_samples
end

-- Create a set of bootstrapped datasets from a table with the original data and
-- a table with the list of bootstrapping samples.
--
-- Each element in the list of boostrap_samples is a list with the elements of
-- the original dataset to be included in that particular bootstrap sample.
--
-- See https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Case_resampling
laia.bootstrap_data = function(data, bootstrap_samples)
  assert(data ~= nil and type(data) == 'table',
	 'laia.bootstrap_data expects an input "data" table')
  assert(bootstrap_samples ~= nil and (type(bootstrap_samples) == 'table' or
			type(bootstrap_samples) == 'cdata' or laia.isint(bootstrap_samples)),
	 'laia.bootstrap_data expects an input "bootstrap_samples" table ' ..
	   'or integer')
  if laia.isint(bootstrap_samples) then
    bootstrap_samples = laia.bootstrap_sample(#data, bootstrap_samples)
  end
  local bootstrap_data = tds.Vec()
  for s=1,#bootstrap_samples do
    if #data ~= #bootstrap_samples[s] then
      laia.log.warn('The number of data items in the bootstrap sample %d ' ..
		      'should be equal to the number of original data items ' ..
		      '(expected = %d, actual = %d)',
		    s, #data, #bootstrap_samples[s])
    end
    bootstrap_data:insert(tds.Vec())
    for _,i in ipairs(bootstrap_samples[s]) do
	bootstrap_data[s]:insert(data[i])  
    end
  end
  return bootstrap_data
end

-------------------------------------------------------------------------------
-- T-STUDENT TEST
-------------------------------------------------------------------------------

-- Compute the t-statistic from two paired samples, for the null hypothesis (H0)
-- "the mean of the differences between the two samples is 0".
-- Note: If there are N paired samples, then the number of degrees of freedom
-- is N - 1.
-- See: https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples
laia.t_statistic_paired = function(data1, data2)
  assert(type(data1) == 'table' and type(data2) == 'table' and #data1 == #data2,
	 ('In order to perform a t-test with paired samples, the number of ' ..
	    'samples in the two datasets must be the same (%d vs %d)'):format(
	   #data1, #data2))
  local N = #data1
  -- Compute the sample mean (m) and standard deviation (s) of the differences
  -- in the paired datasets.
  local acc1, acc2 = 0, 0
  for i=1,N do
    local d = data1[i] - data2[i]
    acc1 = acc1 + d
    acc2 = acc2 + d * d
  end
  local m = acc1 / N
  local s = math.sqrt((acc2 - 2 * N * m * m) / (N - 1))
  -- Return t-statistic for the paired sample t-test (N - 1 degrees of freedom)
  return (m / s) * math.sqrt(N)
end

