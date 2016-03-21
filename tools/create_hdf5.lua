local argparse = require 'argparse';
local image = require 'image';
local hdf5 = require 'hdf5';
local torch = require 'torch';

local str2bool_table = {
   ['true'] = true, ['false'] = false,
   ['t'] = true, ['f'] = false,
   ['True'] = true, ['False'] = false,
   ['1'] = true, ['0'] = false
};

-- Tool options and arguments.
local parser = argparse('crate_hdf5.lua', 'Create a HDF5 file containing ' ..
			'the images and transcriptions of a set of samples ' ..
			'ready to be used by Torch7.');
parser:argument('transcripts', 'File containing the transcript of each ' ..
		'sample, one per line');
parser:argument('symbols_table', 'File containing the mapping between ' ..
		'symbols and integers');
parser:argument('images_dir', 'Directory containing the PNG images of ' ..
		'each of the samples in the transcripts file');
parser:argument('output', 'Output LMDB file containing the samples');
parser:option('-b --binarize', 'Binarize image using simple threshdolding ' ..
		 'algorithm', nil):convert(tonumber);
parser:option('-i --invert', 'Invert colors (0 is white and 1 is black)',
	      'false'):convert(str2bool_table);
parser:option('-H --height',
	      'If > 0, resize samples to have this height and a ' ..
	      'proportional width, keeping the aspecet ratio',
	      0):convert(tonumber);
parser:option('-M --mean',
	      'Subtract this value to the pixel intensity',
	      0.0):convert(tonumber):action(
   function(_, _, mean)
      assert(mean >= 0.0 and mean <= 1.0,
	     string.format('Mean value must be in the range [0.0, 1.0]'));
   end);
parser:option('-S --stddev',
	      'Divide the subtracted pixel intensity by this value',
	      1.0):convert(tonumber):action(
   function(_, _, stddev)
      assert(stddev > 0.0,
	     string.format('Standard deviation value must be greater than 0'));
   end);
local args = parser:parse()

-- Read symbols table file into sym2int
local sym2int = {}
local f = io.open(args.symbols_table, 'r')
assert(f ~= nil, string.format('File %q cannot be read!', args.symbols_table))
local ln = 0
while true do
   local line = f:read('*line');
   if line == nil then break end;
   ln = ln + 1;
   local sym, id = string.match(line, '^(%S+)%s+(%d+)$');
   assert(sym ~= nil and id ~= nil,
	  string.format('Expected a string and a integer separated by ' ..
			'whitespaces at line %d in file %q',
			ln, args.symbols_table));
   sym2int[sym] = tonumber(id);
end;
f:close()

local hf = hdf5.open(args.output, 'w')
local options = hdf5.DataSetOptions()
options:setChunked(64, 64, 64);
options:setDeflate();

-- Process all samples (load images and process transcripts)
local f = io.open(args.transcripts, 'r')
assert(f ~= nil, string.format('File %q cannot be read!', args.symbols_table))
local ln = 0
while true do
   local line = f:read('*line')
   if line == nil then break; end;
   ln = ln + 1
   -- Process sample transcript
   local id, txt = string.match(line, '^(%S+)%s+(%S.*)$')
   assert(id ~= nil and txt ~= nil,
	  string.format('Wrong transcripts format at line %d in file %q',
			ln, args.transcripts))
   txt2int = {}
   for sym in txt:gmatch('%S+') do
      assert(sym2int[sym] ~= nil, string.format(
		'Symbol %q is not found in the symbols table %q',
		sym, args.symbols_table));
      table.insert(txt2int, sym2int[sym]);
   end;
   txt2int = torch.IntTensor(torch.IntStorage(txt2int))
   hf:write(string.format('%s/gt', id), txt2int)
   -- Load sample image
   local img_path = string.format('%s/%s.png', args.images_dir, id)
   local img = image.load(img_path, 1, 'float')
   -- Resize to fixed height
   if args.height > 0 then
      local width = img:size()[3] * args.height / img:size()[2];
      img = image.scale(img, width, args.height)
   end;
   -- Invert colors
   if args.invert == true then
      img:apply(function(v) return 1.0 - v; end);
   end
   -- Normalize image
   img = (img - args.mean) / args.stddev;
   -- Simple threshold binarization
   if args.binarize ~= nil then
      img[img:lt(args.binarize)] = 0.0;
      img[img:ge(args.binarize)] = 1.0;
   end
   hf:write(string.format('%s/img', id), img);
end;
f:close()
hf:close()
