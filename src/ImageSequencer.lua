require 'torch';
require 'rnn';

local ImageSequencer, parent = torch.class('nn.ImageSequencer',
					   'nn.Module');

function ImageSequencer:__init()
   parent.__init(self)
end;

function ImageSequencer:updateOutput(input)
   assert(input:dim() == 4,
	  'Input must be a 4-dimensional tensor (N x D x H x W)')
   input = input:permute(4, 1, 2, 3):contiguous()
   input = input:view(input:size()[1], input:size()[2],
		      input:size()[3] * input:size()[4])
   self.output = {}
   for i=1,input:size()[1] do
      table.insert(self.output, input[i]);
   end;
   return self.output
end;
