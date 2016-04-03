require 'torch';

require 'cudnn';
require 'Model';
require 'Utils';
require 'WidthBatcher';

local str2bool_table = {
   ['true'] = true, ['false'] = false,
   ['t'] = true, ['f'] = false,
   ['True'] = true, ['False'] = false,
   ['1'] = true, ['0'] = false,
   ['TRUE'] = true, ['FALSE'] = false
};

local argparse = require 'argparse';
local parser = argparse('decode.lua', '');
parser:argument('model', 'Path to the neural network model file');
parser:argument('data', 'Path to the dataset HDF5 file');
parser:option('-b --batch_size', 'Batch size', 16):convert(tonumber);
parser:option('--use_gpu',
	      'If true, perform computations on a GPU card supporintg CUDA',
	      'true'):convert(str2bool_table);
parser:option('--use_cudnn',
	      'If true, use NVIDIA cuDNN toolkit',
	      'true'):convert(str2bool_table);
local args = parser:parse();

if args.use_gpu then
   require 'cutorch';
   require 'cunn';
   if args.use_cudnn then require 'cudnn'; end;
end;

local model = torch.load(args.model);
model:remove();  -- Remove last module (JoinTable)

if args.use_gpu then
   model = model:cuda();
   if use_cudnn then cudnn.convert(model, cudnn); end;
else
   model = model:float();
end;

local dv = WidthBatcher(args.data, true);
local n = 0;
for batch=1,dv:numSamples(),args.batch_size do
   -- Prepare batch
   local batch_img, _, _, batch_ids = dv:next(args.batch_size);
   if args.use_gpu then batch_img = batch_img:cuda(); end;
   -- Forward through network
   local output = model:forward(batch_img);
   -- Prepare hypothesis
   local hyps = {};
   for t=1,#output do
      local _, idx = torch.max(output[t], 2);
      idx = torch.totable(idx - 1);
      for i=1,args.batch_size do
	 if i <= #hyps then
	    table.insert(hyps[i], idx[i][1])
	 else
	    hyps[i] = idx[i];
	 end;
      end;
   end;
   for i=1,args.batch_size do
      n = n + 1;
      if n > dv:numSamples() then break; end;
      io.write(string.format('%s  ', batch_ids[i]));
      for t=1,#hyps[i] do
	 io.write(string.format(' %d', hyps[i][t]));
      end;
      io.write('\n');
   end;
end;
