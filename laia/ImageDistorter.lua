require 'imgdistort'
require 'pl'

local ImageDistorter = torch.class('laia.ImageDistorter')

function ImageDistorter:__init()
  self._opt = {
    -- Scale parameters. Scaling is applied at the center of the image.
    scale_prob = 0.5,
    scale_mean = 0.0,
    scale_stdv = 0.12,
    -- Horizontal shear parameters.
    shear_prob = 0.5,
    shear_prec = 4,
    -- Rotate parameters [relative to the maximum aspect ratio of the image].
    rotate_prob = 0.5,
    rotate_prec = 50,
    -- Translate parameters [relative to the size of each dimension].
    translate_prob = 0.5,
    translate_stdv = 0.02,
    -- Dilate parameters.
    dilate_prob = 0.5,
    dilate_srate = 0.4,
    dilate_rrate = 0.8,
    -- Erode parameters.
    erode_prob = 0.5,
    erode_srate = 0.8,
    erode_rrate = 1.0
  }
end

function ImageDistorter:registerOptions(parser)
  -- Scale parameters. Scaling is applied at the center of the image.
  parser:option(
    '--distort_scale_prob',
    'Probability of scaling an image. Scaling is relative to the center of ' ..
      'the image and the scaling factor is sampled from a log-normal ' ..
      'distribution (see --distort_scale_mean and --distort_scale_stdv).',
    self._opt.scale_prob, tonumber)
    :argname('<p>')
    :overwrite(false)
    :ge(0.0):le(1.0)
    :bind(self._opt, 'scale_prob')
  parser:option(
    '--distort_scale_mean', 'Mean of the log of the scaling factor.',
    self._opt.scale_mean, tonumber)
    :argname('<m>')
    :overwrite(false)
    :gt(0.0)
    :bind(self._opt, 'scale_mean')
  parser:option(
    '--distort_scale_stdv', 'Standard deviation of the log of the scaling ' ..
      'factor.', self._opt.scale_stdv, tonumber)
    :argname('<s>')
    :overwrite(false)
    :gt(0.0)
    :bind(self._opt, 'scale_stdv')
  -- Horizontal shear parameters.
  parser:option(
    '--distort_shear_prob',
    'Probability of applying horizontal shear to an image. The shear angle ' ..
      '(in radians) is sampled from a von Mises distribution (see ' ..
      '--distort_shear_prec).', self._opt.shear_prob, tonumber)
    :argname('<p>')
    :overwrite(false)
    :ge(0.0):le(1.0)
    :bind(self._opt, 'shear_prob')
  parser:option(
    '--distort_shear_prec', 'Precision of the shear angle (rad).',
    self._opt.shear_prec, tonumber)
    :argname('<s>')
    :overwrite(false)
    :gt(0.0)
    :bind(self._opt, 'shear_prec')
  -- Translate parameters. Translation is relative to the size in each
  -- dimension.
  parser:option(
    '--distort_translate_prob',
    'Probability of applying a translation to an image. The translation is ' ..
      'relative to the dimension size and it is sampled from a normal ' ..
      'distribution (see --distort_translate_stdv).', self._opt.translate_prob,
    tonumber)
    :argname('<p>')
    :overwrite(false)
    :ge(0.0):le(1.0)
    :bind(self._opt, 'translate_prob')
  parser:option(
    '--distort_translate_stdv',
    'Standard deviation of the translation across each dimension ' ..
      'actual stdv is s\' = s * dimension_size.', self._opt.translate_stdv,
    tonumber)
    :argname('<s>')
    :gt(0.0)
    :overwrite(false)
    :bind(self._opt, 'translate_stdv')
  -- Rotate parameters. Rotation is applied at the center of the image.
  -- Rotation is relative to the maximum aspect ratio, i.e: max(W/H, H/W).
  parser:option(
    '--distort_rotate_prob',
    'Probability of applying a rotation to an image. The rotation angle (in ' ..
      'radians) is relative to the maximum aspect ratio of the image and it ' ..
      'is sampled from a von Mises distribution (see --distort_rotate_prec).',
    self._opt.rotate_prob, tonumber)
    :argname('<p>')
    :overwrite(false)
    :ge(0.0):le(1.0)
    :bind(self._opt, 'rotate_prob')
  parser:option(
    '--distort_rotate_prec',
    'Precision of the rotation angle (rad), actual used precision is ' ..
      's\' = s * max(H/W, W/H).', self._opt.rotate_prec, tonumber)
    :argname('<s>')
    :ge(0.0)
    :overwrite(false)
    :bind(self._opt, 'rotate_prec')
  -- Dilate parameters.
  parser:option(
    '--distort_dilate_prob',
    'Probability of applying a dilate distortion to an image. The dilate ' ..
      'kernel is built from two distributions (see --distort_dilate_srate ' ..
      'and --distort_dilate_rrate).', self._opt.dilate_prob, tonumber)
    :argname('<p>')
    :ge(0.0):lt(1.0)
    :overwrite(false)
    :bind(self._opt, 'dilate_prob')
  parser:option(
    '--distort_dilate_srate',
    'Rate of the geometric distribution used to sample the size of the size ' ..
      'of the dilate kernel. Possible kernel sizes are: 3, 5, 7, 9, 11, 15. ' ..
      'If the rate is r, the probability of selecting the size s is ' ..
      'proportional to r * (1 - r)^(s - 3).', self._opt.dilate_srate, tonumber)
    :argname('<r>')
    :ge(0.0):le(1.0)
    :overwrite(false)
    :bind(self._opt, 'dilate_srate')
  parser:option(
    '--distort_dilate_rrate',
    'An element of the dilate kernel at an euclidean distance of d from the ' ..
      'center of the kernel is set to 1 with a probability equal to ' ..
      'exp(-d * r).', self._opt.dilate_rrate, tonumber)
    :argname('<r>')
    :ge(0.0)
    :overwrite(false)
    :bind(self._opt, 'dilate_rrate')
  -- Erode parameters.
  parser:option(
    '--distort_erode_prob',
    'Probability of applying an erode distortion to an image. The erode ' ..
      'kernel is built from two distributions (see --distort_erode_srate ' ..
      'and --distort_erode_rrate).', self._opt.erode_prob, tonumber)
    :argname('<p>')
    :ge(0.0):lt(1.0)
    :overwrite(false)
    :bind(self._opt, 'erode_prob')
  parser:option(
    '--distort_erode_srate',
    'Rate of the geometric distribution used to sample the size of the size ' ..
      'of the erode kernel. Possible kernel sizes are: 3, 5, 7, 9, 11, 15. ' ..
      'If the rate is r, the probability of selecting the size s is ' ..
      'proportional to (1 - r)^(s - 3) * r.', self._opt.erode_srate, tonumber)
    :argname('<r>')
    :ge(0.0):le(1.0)
    :overwrite(false)
    :bind(self._opt, 'erode_srate')
  parser:option(
    '--distort_erode_rrate',
    'An element of the erode kernel at an euclidean distance of d from the ' ..
      'center of the kernel is set to 1 with a probability equal to ' ..
      'exp(-d * r).', self._opt.erode_rrate, tonumber)
    :argname('<r>')
    :ge(0.0)
    :overwrite(false)
    :bind(self._opt, 'erode_rrate')
end

function ImageDistorter:setOptions(opt)
  table.update_values(self._opt, opt)
  self:checkOptions()
end

function ImageDistorter:checkOptions()
  -- Scale parameters.
  assert(self._opt.scale_prob >= 0.0 and self._opt.scale_prob <= 1.0)
  assert(self._opt.scale_stdv > 0.0)
  -- Horizontal shear parameters.
  assert(self._opt.shear_prob >= 0.0 and self._opt.shear_prob <= 1.0)
  assert(self._opt.shear_prec >= 0.0)
  -- Rotate parameters.
  assert(self._opt.rotate_prob >= 0.0 and self._opt.rotate_prob <= 1.0)
  assert(self._opt.rotate_prec >= 0.0)
  -- Translate parameters.
  assert(self._opt.translate_prob >= 0.0 and self._opt.translate_prob <= 1.0)
  assert(self._opt.translate_stdv > 0.0)
  -- Dilate parameters.
  assert(self._opt.dilate_prob >= 0.0 and self._opt.dilate_prob <= 1.0)
  assert(self._opt.dilate_srate >= 0.0 and self._opt.dilate_srate <= 1.0)
  assert(self._opt.dilate_rrate >= 0.0)
  -- Erode parameters.
  assert(self._opt.erode_prob >= 0.0  and self._opt.erode_prob <= 1.0)
  assert(self._opt.erode_srate >= 0.0 and self._opt.erode_srate <= 1.0)
  assert(self._opt.erode_rrate >= 0.0)
end

function ImageDistorter:distort(x, sizes, y)
  assert(x:nDimension() == 4, 'Input to ImageDistorter must be a 4-dim ' ..
	   'tensor with NCHW layout.')
  local x = x:clone():cuda()
  y = y and y:resizeAs(x):cuda():zero() or x:clone():zero()
  -- Affine distortion
  local N, H, W = x:size()[1], x:size()[3], x:size()[4]
  local M = self:__sample_affine_matrixes(N, H, W, sizes)
  affine_NCHW(x, y, M)
  -- Morphology distortion
  if self._opt.dilate_prob > 0 then
    M = self:__sample_structuring_element(
      N, self._opt.dilate_prob, self._opt.dilate_srate, self._opt.dilate_rrate)
    x, y = y, x
    dilate_NCHW(x, y, M)
  end
  if self._opt.erode_prob > 0 then
    M = self:__sample_structuring_element(
      N, self._opt.erode_prob, self._opt.erode_srate, self._opt.erode_rrate)
    x, y = y, x
    erode_NCHW(x, y, M)
  end
  return y
end

function ImageDistorter:__sample_affine_matrixes(N, H, W, sizes)
  local T = torch.zeros(N, 2, 3)
  -- Matrix used to center the transformation relative to the center of
  -- the image: (x, y) = (W / 2, H / 2)
  local C, Cm = torch.eye(3), torch.eye(3)

  for i=1,N do
    if sizes then
      C[{1, 3}] = sizes[i][2] / 2    Cm[{1, 3}] = - sizes[i][2] / 2
      C[{2, 3}] = sizes[i][1] / 2    Cm[{2, 3}] = - sizes[i][1] / 2
    else
      C[{1, 3}] = W / 2              Cm[{1, 3}] = - W / 2
      C[{2, 3}] = H / 2              Cm[{2, 3}] = - H / 2
    end
    local Ti = T[{i}]:eye(3)
    -- Matrixes have to be multiplied in reverse order of the logical
    -- operations: First translate, then rotate, then shear and finally scale
    if torch.uniform() < self._opt.translate_prob then
      local D =	torch.randn(2) * self._opt.translate_stdv
      Ti[{1, 3}] = D[1] * W
      Ti[{2, 3}] = D[2] * H
    end
    if torch.uniform() < self._opt.rotate_prob then
      local D = torch.eye(3)
      local R = math.max((sizes and sizes[i][2] / sizes[i][1]) or W / H,
			 (sizes and sizes[i][1] / sizes[i][2]) or H / W)
      local prec = R * self._opt.rotate_prec
      local a = laia.rand_von_Mises(0.0, prec) or 0
      D[{1, 1}] =  math.cos(a)
      D[{1, 2}] = -math.sin(a)
      D[{2, 1}] =  math.sin(a)
      D[{2, 2}] =  math.cos(a)
      Ti:copy(torch.mm(torch.mm(torch.mm(Ti, C), D), Cm))
    end
    if torch.uniform() < self._opt.shear_prob then
      local a = laia.rand_von_Mises(0.0, self._opt.shear_prec)
      local D = torch.eye(3, 3)
      D[{1, 2}] = a
      Ti:copy(torch.mm(torch.mm(torch.mm(Ti, C), D), Cm))
    end
    if torch.uniform() < self._opt.scale_prob then
      local f = math.exp(torch.randn(1)[1] * self._opt.scale_stdv +
			   self._opt.scale_mean)
      local D = torch.eye(3)
      D[{1, 1}] = f
      D[{2, 2}] = f
      Ti:copy(torch.mm(torch.mm(torch.mm(Ti, C), D), Cm))
    end
  end
  return T:cuda()
end

function ImageDistorter:__sample_structuring_element(N, p, srate, rrate)
  -- Compute size of the structuring element. Only the following sizes are
  -- valid, to simplify the implementation.
  local Sv = {3, 5, 7, 9, 11, 13, 15}
  local Sp = table.map(
    Sv, function(x) return srate * torch.pow(1 - srate, x - Sv[1]) end)
  local Mh = Sv[table.weighted_choice(Sp, Sz)]  -- Sample kernel height
  local Mw = Sv[table.weighted_choice(Sp, Sz)]  -- Sample kernel width
  -- Compute radius likelihoods
  local M = torch.ByteTensor(N, Mh, Mw):zero()
  for n=1,N do
    if torch.uniform() < p then
      for y=0,(Mh-1) do
	for x=0,(Mw-1) do
	  local dy = y - math.floor(Mh / 2)
	  local dx = x - math.floor(Mw / 2)
	  local r  = math.sqrt(dx * dx + dy * dy)
	  M[{n, y + 1, x + 1}] =
	    (torch.uniform() < math.exp(-rrate * r) and 1) or 0
	end
      end
    else
      M[{n, 1 + math.floor(Mh / 2), 1 + math.floor(Mw / 2)}] = 1
    end
  end
  return M:type('torch.CudaByteTensor')
end

return ImageDistorter
