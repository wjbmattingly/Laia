require 'cutorch'
require 'imgdistort'

local ImageDistorter = torch.class('ImageDistorter')

function ImageDistorter:__init()
   -- Scale parameters. Scaling is applied at the center of the image.
   self.scale_prob = 0.5
   self.scale_mean = 1.0
   self.scale_stdv = 0.15

   -- Horizontal shear parameters.
   self.shear_prob = 0.5
   self.shear_mean = 0.0
   self.shear_stdv = 0.15

   -- Rotate parameters. Rotation is applied at the center of the image.
   self.rotate_prob = 0.5
   self.rotate_mean = 0.0
   self.rotate_stdv = math.rad(5.0)

   -- Translate parameters. Standard deviation is relative to the size of
   -- each dimension.
   self.translate_prob = 0.5
   self.translate_mean = 0.0
   self.translate_stdv = 0.05
end

function ImageDistorter:__sample_affine_matrixes(N, H, W)
   local T = torch.zeros(N, 2, 3)
   -- Matrix used to center the transformation to (x, y) = (W / 2, H / 2)
   local C, Cm = torch.eye(3), torch.eye(3)
   C[{1, 3}] = W / 2    Cm[{1, 3}] = - W / 2
   C[{2, 3}] = H / 2    Cm[{2, 3}] = - H / 2
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
	 local a = torch.randn(1)[1] * self.rotate_stdv + self.rotate_mean
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

function ImageDistorter:distort(x)
   assert(x:nDimension() == 4, 'Input to ImageDistorter must be a 4-dim ' ..
	  'tensor with NCHW layout.')
   local N, H, W = x:size()[1], x:size()[3], x:size()[4]
   local M = self:__sample_affine_matrixes(N, H, W)
   x = x:cuda()
   local y = x:clone():zero()
   affine_NCHW(x, y, M)
   return y
end
