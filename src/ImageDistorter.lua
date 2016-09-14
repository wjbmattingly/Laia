require 'cutorch'
require 'imgdistort'
require 'src.utilities'

local ImageDistorter = torch.class('ImageDistorter')

function ImageDistorter:__init(...)
   local arg = {...}
   -- Scale parameters. Scaling is applied at the center of the image.
   self.scale_prob = 0.5
   if arg.scale_prob ~= nil then self.scale_prob = arg.scale_prob end
   self.scale_mean = 1.0
   if arg.scale_mean ~= nil then self.scale_mean = arg.scale_mean end
   self.scale_stdv = 0.12
   if arg.scale_stdv ~= nil then self.scale_stdv = arg.scale_stdv end

   -- Horizontal shear parameters.
   self.shear_prob = 0.5
   if arg.shear_prob ~= nil then self.shear_prob = arg.shear_prob end
   self.shear_mean = 0.0
   if arg.shear_mean ~= nil then self.shear_mean = arg.shear_mean end
   self.shear_stdv = 0.5
   if arg.shear_stdv ~= nil then self.shear_stdv = arg.shear_stdv end

   -- Rotate parameters. Rotation is applied at the center of the image.
   -- The standard deviation depends on the size of the image, so that the
   -- farest points from the center are not moved too much.
   self.rotate_prob = 0.5
   if arg.rotate_prob ~= nil then self.rotate_prob = arg.rotate_prob end
   self.rotate_factor = 0.4
   if arg.rotate_factor ~= nil then self.rotate_factor = arg.rotate_factor end

   -- Translate parameters. Standard deviation is relative to the size of
   -- each dimension.
   self.translate_prob = 0.5
   if arg.translate_prob ~= nil then self.translate_prob=arg.translate_prob end
   self.translate_mean = 0.0
   if arg.translate_mean ~= nil then self.translate_mean=arg.translate_mean end
   self.translate_stdv = 0.02
   if arg.translate_stdv ~= nil then self.translate_stdv=arg.translate_stdv end

   -- Dilate parameters.
   self.dilate_prob = 0.5
   if arg.dilate_prob ~= nil then self.dilate_prob = arg.dilate_prob end
   self.dilate_srate = 0.5
   if arg.dilate_srate ~= nil then self.dilate_srate = arg.dilate_srate end
   self.dilate_rrate = 0.8
   if arg.dilate_rrate ~= nil then self.dilate_rrate = arg.dilate_rrate end

   -- Erode parameters.
   self.erode_prob = 0.5
   if arg.erode_prob ~= nil then self.erode_prob = arg.erode_prob end
   self.erode_srate = 0.5
   if arg.erode_srate ~= nil then self.erode_srate = arg.erode_srate end
   self.erode_rrate = 0.8
   if arg.erode_rrate ~= nil then self.erode_rrate = arg.erode_rrate end
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
