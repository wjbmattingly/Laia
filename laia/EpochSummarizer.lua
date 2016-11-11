require 'laia.ClassWithOptions'
require 'laia.Statistics'

local EpochSummarizer, Parent = torch.class('laia.EpochSummarizer',
					    'laia.ClassWithOptions')

function EpochSummarizer:__init()
  Parent.__init(self, {
    bootstrap_confidence_interval = true,
    bootstrap_num_samples = 1000,
    bootstrap_alpha = 0.05,
  })
end

function EpochSummarizer:registerOptions(parser, advanced)
  advanced = advanced or false
  parser:option(
    '--epoch_confidence_interval',
    'If true, compute percentile bootstrap confidence intervals for the CER ' ..
      'after each epoch.',
    self._opt.bootstrap_confidence_interval, laia.toboolean)
    :argname('<bool>')
    :bind(self._opt, 'bootstrap_confidence_interval')
    :advanced(advanced)
  parser:option(
    '--epoch_bootstrap_samples',
    'If n>0, uses n bootstrap samples. If n=0, the number of bootstrap ' ..
      'samples is equal to the number of train/valid samples.',
    self._opt.bootstrap_num_samples, laia.toint)
    :argname('<n>')
    :ge(0)
    :bind(self._opt, 'bootstrap_num_samples')
    :advanced(advanced)
  parser:option(
    '--epoch_bootstrap_alpha',
    'Use this alpha value to compute the confidence intervals. E.g. ' ..
      'alpha = 0.05 means a confidence interval at 95%',
    self._opt.bootstrap_alpha, tonumber)
    :argname('<alpha>')
    :gt(0):lt(0.5)
    :bind(self._opt, 'bootstrap_alpha')
    :advanced(advanced)
end

function EpochSummarizer:checkOptions()
  assert(laia.isboolean(self._opt.bootstrap_confidence_interval))
  assert(laia.isint(self._opt.bootstrap_num_samples) and
	   self._opt.bootstrap_num_samples >= 0)
  assert(self._opt.bootstrap_alpha > 0 and self._opt.bootstrap_alpha < 0.5)
end

function EpochSummarizer:summarize(info)
  if not info then return nil end
  local loss = info.loss / info.num_frames
  local num_ins_ops = table.reduce(info.num_ins_ops, operator.add, 0)
  local num_del_ops = table.reduce(info.num_del_ops, operator.add, 0)
  local num_sub_ops = table.reduce(info.num_sub_ops, operator.add, 0)
  local ref_length  = table.reduce(info.ref_trim,
				   function(acc, x) return acc + #x end, 0)
  local err = (num_ins_ops + num_del_ops + num_sub_ops) / ref_length
  local duration = os.difftime(info.time_end, info.time_start)
  -- Summary to return
  local summary = {
    loss = loss,
    cer = err,
    ins_ops = num_ins_ops / ref_length,
    del_ops = num_del_ops / ref_length,
    sub_ops = num_sub_ops / ref_length,
    duration = duration
  }
  -- Compute bootstrap confidence intervals for the CER:
  if self._opt.bootstrap_confidence_interval then
    local sample_size = #info.ref_trim
    local sample_edit_ops = {}
    local sample_ref_len  = {}
    for n=1,sample_size do
      table.insert(
	sample_edit_ops,
	info.num_ins_ops[n] + info.num_del_ops[n] + info.num_sub_ops[n])
      table.insert(sample_ref_len,  #info.ref_trim[n])
    end
    local boot_num_samples = self._opt.bootstrap_num_samples > 0 and
      self._opt.bootstrap_num_samples or sample_size
    local boot_samples = laia.bootstrap_sample(sample_size, boot_num_samples)
    local boot_edit_ops = laia.bootstrap_data(sample_edit_ops, boot_samples)
    local boot_ref_len  = laia.bootstrap_data(sample_ref_len, boot_samples)
    local boot_err_diffs = {}
    for s=1,boot_num_samples do
      local s_err = table.reduce(boot_edit_ops[s], operator.add, 0) /
	table.reduce(boot_ref_len[s], operator.add, 0)
      table.insert(boot_err_diffs, s_err - err)
    end
    table.sort(boot_err_diffs)
    -- Compute bootstrap confidence intervals for the population of errors.
    -- Watch that lower/upper bounds are obtained from 97.5%/2.5%, respectively!
    local lalpha = 1.0 - self._opt.bootstrap_alpha / 2
    local ualpha = self._opt.bootstrap_alpha / 2
    local err_lower = err - boot_err_diffs[math.ceil(lalpha * #boot_err_diffs)]
    local err_upper = err - boot_err_diffs[math.ceil(ualpha * #boot_err_diffs)]
    summary.cer_ci = {
      lower = err_lower, upper = err_upper, alpha = self._opt.bootstrap_alpha
    }
  end
  return summary
end

function EpochSummarizer.ToString(summary)
  local format = 'duration = %6s ; loss = %8.6f ; cer = %6.2f%% ; ' ..
    'del = %6.2f%% ; ins = %6.2f%% ; sub = %6.2f%%'
  if summary.cer_ci then
    format = format .. ' ; cer_ci = [%6.2f%%,%6.2f%%] ; ci_alpha = %6.3f%%'
    return format:format(laia.sec_to_dhms(summary.duration),
			 summary.loss,
			 summary.cer * 100,
			 summary.del_ops * 100,
			 summary.ins_ops * 100,
			 summary.sub_ops * 100,
			 summary.cer_ci.lower * 100,
			 summary.cer_ci.upper * 100,
			 summary.cer_ci.alpha * 100)
  else
    return format:format(laia.sec_to_dhms(summary.duration),
			 summary.loss,
			 summary.cer * 100,
			 summary.del_ops * 100,
			 summary.ins_ops * 100,
			 summary.sub_ops * 100)
  end
end

return EpochSummarizer
