require 'cutorch'
require 'imgdistort'
require 'src.utilities'

local ImageDistorter = torch.class('ImageDistorter')

function ImageDistorter:__init(opts)
  -- Scale parameters. Scaling is applied at the center of the image.
  self.scale_prob = 0.5
  self.scale_mean = 1.0
  self.scale_stdv = 0.1

  -- Horizontal shear parameters.
  self.shear_prob = 0.5
  self.shear_mean = 0.0
  self.shear_stdv = 0.3

  -- Rotate parameters. Rotation is applied at the center of the image.
  -- The standard deviation depends on the size of the image, so that the
  -- farest points from the center are not moved too much.
  self.rotate_prob = 0.5
  self.rotate_factor = 0.3

  -- Translate parameters. Standard deviation is relative to the size of
  -- each dimension.
  self.translate_prob = 0.5
  self.translate_mean = 0.0
  self.translate_stdv = 0.01

  -- Dilate parameters.
  self.dilate_prob = 0.5
  self.dilate_srate = 0.5
  self.dilate_rrate = 1.0

  -- Erode parameters.
  self.erode_prob = 0.5
  self.erode_srate = 0.5
  self.erode_rrate = 1.0

  self:parseCmdOptions(opts)
end

function ImageDistorter:__sample_affine_matrixes(N, H, W)
  local T = torch.zeros(N, 2, 3)
  local R = (H > W and W / H) or H / W
  -- Matrix used to center the transformation to (x, y) = (W / 2, H / 2)
  local C, Cm = torch.eye(3), torch.eye(3)
  C[{1, 3}] = W / 2    Cm[{1, 3}] = - W / 2
  C[{2, 3}] = H / 2    Cm[{2, 3}] = - H / 2

  local max_rotate_h = H * self.rotate_factor / math.sqrt(H * H + W * W)
  local max_rotate_w = W * self.rotate_factor / math.sqrt(H * H + W * W) - 1.0
  local rotate_stdv = math.pi
  if math.abs(max_rotate_h) <= 1.0 and math.abs(max_rotate_w) <= 1.0 then
    rotate_stdv = math.min(math.asin(max_rotate_w) + math.pi / 2.0,
                           math.asin(max_rotate_h))
  elseif math.abs(max_rotate_h) <= 1.0 then
    rotate_stdv = math.asin(max_rotate_h)
  elseif math.abs(max_rotate_w) <= 1.0 then
    rotate_stdv = math.asin(max_rotate_w) + math.pi / 2.0
  end

  for i=1,N do
    local Ti = T[{i}]
    Ti:eye(3)
    -- Matrixes have to be multiplied in reverse order of the logical
    -- operations: First translate, then rotate, then shear and finally scale
    if torch.uniform() < self.translate_prob then
      local D = torch.eye(3)
      D[{1, 3}] =
        torch.randn(1)[1] * self.translate_stdv * W + self.translate_mean
      D[{2, 3}] =
        torch.randn(1)[1] * self.translate_stdv * H + self.translate_mean
      Ti:copy(torch.mm(Ti, D))
    end
    if torch.uniform() < self.rotate_prob then
      local D = torch.eye(3)
      local a = torch.randn(1)[1] * rotate_stdv
      D[{1, 1}] =  math.cos(a)
      D[{1, 2}] = -math.sin(a)
      D[{2, 1}] =  math.sin(a)
      D[{2, 2}] =  math.cos(a)
      Ti:copy(torch.mm(torch.mm(torch.mm(Ti, C), D), Cm))
    end
    if torch.uniform() < self.shear_prob then
      local a = torch.randn(1)[1] * self.shear_stdv + self.shear_mean
      local D = torch.eye(3, 3)
      D[{1, 2}] = a
      Ti:copy(torch.mm(torch.mm(torch.mm(Ti, C), D), Cm))
    end
    if torch.uniform() < self.scale_prob then
      local f = torch.randn(1)[1] * self.scale_stdv + self.scale_mean
      local D = torch.eye(3)
      D[{1, 1}] = f
      D[{2, 2}] = f
      Ti:copy(torch.mm(torch.mm(torch.mm(Ti, C), D), Cm))
    end
  end
  return T:cuda()
end

function ImageDistorter:__sample_structuring_element(N, p, srate, rrate)
  assert(p >= 0.0 and p <= 1.0,
         'Dilate/Erode probability must be in the range [0, 1]!')
  assert(srate > 0.0 and srate < 1.0,
         'Size rate value must be in the range (0, 1)!')
  assert(rrate > 0.0, 'Radius rate value must be greater than 0!')
  -- Compute size of the structuring element. Only the following sizes are
  -- valid, to simplify the implementation.
  local Sv = {3, 5, 7, 9, 11, 13, 15}
  local Sp = table.map(Sv, function(x)
			        return srate * torch.pow(1 - srate, x - 1)
                           end)
  local Sz = table.reduce(Sp, operator.add, 0.0)
  local Mh = Sv[table.weighted_choice(Sp, Sz)]        -- Kernel height
  local Mw = Sv[table.weighted_choice(Sp, Sz)]        -- Kernel width
  -- Compute radius likelihoods
  local M = torch.ByteTensor(N, Mh, Mw):zero()
  for n=1,N do
    if math.random() < p then
      for y=0,(Mh-1) do
        for x=0,(Mw-1) do
          local dy = y - math.floor(Mh / 2)
          local dx = x - math.floor(Mw / 2)
          local r  = math.sqrt(dx * dx + dy * dy)
          M[{n, y + 1, x + 1}] =
            (math.random() < math.exp(-rrate * r) and 1) or 0
        end
      end
    else
      M[{n, 1 + math.floor(Mh / 2), 1 + math.floor(Mw / 2)}] = 1
    end
  end
  return M:type('torch.CudaByteTensor')
end

function ImageDistorter:distort(x)
   assert(x:nDimension() == 4, 'Input to ImageDistorter must be a 4-dim ' ..
	  'tensor with NCHW layout.')
   local x = x:clone():cuda()
   local y = x:clone():zero()
   -- Affine distortion
   local N, H, W = x:size()[1], x:size()[3], x:size()[4]
   local M = self:__sample_affine_matrixes(N, H, W)
   affine_NCHW(x, y, M)
   -- Morphology distortion
   if self.dilate_prob > 0 then
      M = self:__sample_structuring_element(N,
					    self.dilate_prob,
					    self.dilate_srate,
					    self.dilate_rrate)
      x, y = y, x
      dilate_NCHW(x, y, M)
   end
   if self.erode_prob > 0 then
      M = self:__sample_structuring_element(N,
					    self.erode_prob,
					    self.erode_srate,
					    self.erode_rrate)
      x, y = y, x
      erode_NCHW(x, y, M)
   end

   return y
end

function ImageDistorter.addCmdOptions(cmd)
  cmd:option('-scale_prob', 0.5, 'Scaling is applied to this ratio of images')
  cmd:option('-scale_mean', 1.0, 'Mean of the scaling factor')
  cmd:option('-scale_stdv', 0.1, 'Standard deviation of the scaling factor')

  cmd:option('-shear_prob', 0.5, 'Shearing is applied to this ratio of images')
  cmd:option('-shear_mean', 0.0, 'Mean of the shearing angle')
  cmd:option('-shear_stdv', 0.3, 'Standard deviation of the shearing angle')

  cmd:option('-rotate_prob', 0.5, 'Rotation is applied to this ratio of images')
  cmd:option('-rotate_factor', 0.3, 'Rotation factor (the higher, the more ' ..
             'rotation is applied')

  cmd:option('-translate_prob', 0.5, 'Translation is applied to this ratio ' ..
             'of images')
  cmd:option('-translate_mean', 0.0, 'Mean of the translation')
  cmd:option('-translate_stdv', 0.01, 'Standard deviation of the translation')

  cmd:option('-dilate_prob', 0.5, 'Dilation is applied to this ratio of images')
  cmd:option('-dilate_srate', 0.5, 'Dilation size follows a geometric ' ..
             'distribution with this rate')
  cmd:option('-dilate_rrate', 1.0, 'Dilation radius follows a geometric ' ..
             'distribution with this rate')

  cmd:option('-erode_prob', 0.5, 'Erosion is applied to this ratio of images')
  cmd:option('-erode_srate', 0.5, 'Erosion size follows a geometric ' ..
             'distribution with this rate')
  cmd:option('-erode_rrate', 1.0, 'Erosion radius follows a geometric ' ..
             'distribution with this rate')
end

function ImageDistorter:parseCmdOptions(opts)
  for k,v in pairs(opts) do
    self[k] = v
  end
end
