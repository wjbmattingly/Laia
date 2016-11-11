require 'laia.util.types'
cudnn = require 'cudnn'
cudnn.force_convert = true

-- Function to register cudnn options to the given parser.
function cudnn.registerOptions(parser, advanced)
  advanced = advanced or false
  parser:option(
    '--cudnn_benchmark',
    'If true, the in-built cudnn auto-tuner is used to find the fastest ' ..
      'convolution algorithms. If false, heuristics are used instead.',
    cudnn.benchmark, laia.toboolean)
    :bind(cudnn, 'benchmark')
    :advanced(advanced)
  parser:option(
    '--cudnn_fastest',
    'If true, picks the fastest convolution algorithm, rather than tuning ' ..
    'for workspace size.', cudnn.fastest, laia.toboolean)
    :bind(cudnn, 'fastest')
    :advanced(advanced)
  parser:option(
    '--cudnn_verbose',
    'If true, prints to stdout verbose information about the cudnn ' ..
      'benchmark algorithm.', cudnn.verbose, laia.toboolean)
    :bind(cudnn, 'verbose')
    :advanced(advanced)
  parser:option(
    '--cudnn_force_convert',
    'If true, tries to use the cudnn implementation for all possible ' ..
      'layers. WARNING: Some cudnn layers produce non-deterministic ' ..
      'results in the backward pass.', cudnn.force_convert, laia.toboolean)
    :bind(cudnn, 'force_convert')
    :advanced(advanced)
end

return cudnn
