require 'cutorch'
require 'torch'
require 'image'
require 'utilities'

local CachedBatcher = torch.class('CachedBatcher')

function CachedBatcher:__init(img_list, cfg)
  self._centered_patch = cfg.centered_patch or true
  self._cache_max_size = cfg.cache_max_size or 512
  self._cache_gpu = cfg.cache_gpu or -1
  self._invert = cfg.invert or 1
  self._min_width = cfg.min_width or 0
  self._imglist = {}
  self._has_gt = cfg.gt_file and true or false
  self._gt = {}
  self._samples = {}
  self._num_samples = 0
  self._idx = 0
  self._cache_img = {}
  self._cache_gt = {}
  self._cache_idx = 0
  self._cache_size = 0

  -- Load sample transcripts
  if self._has_gt then
    -- Load symbols list
    assert(cfg.symbols_table ~= nil, string.format('A symbols list is required when providing transcripts'))
    --local sym2int = read_symbols_table(cfg.symbols_table)
    --assert(#sym2int > 0, string.format('Unable to read symbols file: %q', cfg.symbols_table))
    local sym2int = {}
    local f = io.open(cfg.symbols_table, 'r')
    assert(f ~= nil, string.format('Unable to read symbols file: %q', cfg.symbols_table))
    local ln = 0
    while true do
      local line = f:read('*line')
      if line == nil then break end
      ln = ln + 1
      local sym, id = string.match(line, '^(%S+)%s+(%d+)$')
      assert(sym ~= nil and id ~= nil, string.format('Expected a string and an integer separated by a space at line %d in file %q', ln, cfg.symbols_table))
      sym2int[sym] = tonumber(id)
    end
    f:close()

    -- Load sample transcripts
    local f = io.open(cfg.gt_file, 'r')
    assert(f ~= nil, string.format('Unable to read transcripts file: %q', cfg.gt_file))
    local ln = 0
    while true do
      local line = f:read('*line')
      if line == nil then break end
      ln = ln + 1
      local id, txt = string.match(line, '^(%S+)%s+(%S.*)$')
      assert(id ~= nil and txt ~= nil, string.format('Wrong transcript format at line %d in file %q',
      ln, cfg.gt_file))
      txt2int = {}
      for sym in txt:gmatch('%S+') do
        assert(sym2int[sym] ~= nil, string.format('Symbol %q is not in the symbols table', sym))
        table.insert(txt2int, sym2int[sym])
      end
      self._gt[id] = torch.totable(torch.IntTensor(torch.IntStorage(txt2int)))
    end
    f:close()
  end

  -- Load image list and sample IDs. By default, they are ordered as read from the list.
  -- The IDs are the basenames of the files, i.e., removing directory and extension
  local f = io.open(img_list, 'r')
  assert(f ~= nil, string.format('Unable to read image list file: %q', img_file))
  local ln = 0
  while true do
    local line = f:read('*line')
    if line == nil then break end
    ln = ln + 1
    local id = string.match( string.gsub(line, ".*/", ""), '^(.+)[.][^./]+$' );
    assert(id ~= nil, string.format('Unable to determine sample ID at line %d in file %q', ln, img_list))
    if self._has_gt then
      assert(self._gt[id] ~= nil, string.format('No transcript found for sample %q', id))
    end
    table.insert(self._imglist, line)
    table.insert(self._samples, id)
    self._num_samples = self._num_samples + 1
    if not self._has_gt then
      self._gt[id] = torch.totable(torch.IntTensor(torch.IntStorage({})))
    end
  end
  f:close()
end

function CachedBatcher:numSamples()
  return self._num_samples
end

function CachedBatcher:clearCache()
  self._cache_img = {}
  self._cache_gt = {}
  self._cache_idx = 0
  self._cache_size = 0
end

function CachedBatcher:_global2cacheIndex(idx)
  if #self._cache_img < 1 then return -1 end
  if idx < 1 or idx > self._num_samples then return -1 end
  idx = idx - 1
  local cacheEnd = (self._cache_idx + #self._cache_img) % self._num_samples
  if self._cache_idx < cacheEnd then
    -- Cache is contiguous: [i, i+1, ..., i + M]
    if idx >= self._cache_idx and idx < cacheEnd then
   return 1 + idx - self._cache_idx
    else
   return -1
    end
  else
    -- The cache is circular: [i, i + 1, ..., i + N, 0, ..., M - N - 1]
    if idx >= self._cache_idx or idx < cacheEnd then
     return 1 + (idx - self._cache_idx) % self._num_samples
    else
     return -1
    end
  end
end

function CachedBatcher:_cache2globalIndex(idx)
  if idx < 1 or idx > #self._cache_img then return -1 end
  return 1 + (self._cache_idx + idx - 1) % self._num_samples
end

function CachedBatcher:_fillCache(idx)
  -- If the requested image is not available in the cache, fill the cache
  -- with it and all the following images until the cache size increases up to
  -- self._cache_max_size MB (approx)
  if self:_global2cacheIndex(idx) < 0 then
    idx = idx - 1
    self._cache_img = {}
    self._cache_gt = {}
    self._cache_size = 0
    self._cache_idx = idx  -- First cached image = first in the batch
    -- Allocate memory in the GPU device self._cache_gpu, the current device
    -- is saved and restored later.
    local old_gpu = -1
    if self._cache_gpu >= 0 then
      old_gpu = cutorch.getDevice()
      cutorch.setDevice(self._cache_gpu)
    end
    while self._cache_size < self._cache_max_size and
          #self._cache_img < self._num_samples do
      local j = 1 + (#self._cache_img + idx) % self._num_samples

      -- Add ground truth to cache
      --if self._has_gt then
        table.insert(self._cache_gt, self._gt[self._samples[j]])
      --end
      -- Add image to cache
      local img = image.load(self._imglist[j], 1, 'float')
      -- Invert colors
      if self._invert == 1 or ( self._invert and math.random() > self._invert ) then
        img:apply(function(v) return 1.0 - v end)
      end
      -- Normalize image
      --img = (img - args.mean) / args.stddev
      table.insert(self._cache_img, img)

      -- Store data in the GPU, if requested.
      if self._cache_gpu >= 0 then
        self._cache_img[#self._cache_img] = self._cache_img[#self._cache_img]:cuda()
      end
      -- Increase size of the cache (in MB)
      self._cache_size = self._cache_size +
        self._cache_img[#self._cache_img]:storage():size() * 4 / 1048576 +
        #(self._cache_gt[#self._cache_gt]) * 8 / 1048576
    end
    -- Restore value of the current GPU device
    if old_gpu >= 0 then
      cutorch.setDevice(old_gpu)
    end
  end
end

function CachedBatcher:next(batch_size)
  batch_size = batch_size or 1
  assert(batch_size > 0, string.format('Batch size must be greater than 0'))
  assert(self._num_samples > 0, string.format('The dataset is empty!'))
  -- Get sizes of each sample in the batch
  local n_channels = nil
  local batch_sizes = torch.LongTensor(batch_size, 2)
  for i=0,(batch_size-1) do
    local j = 1 + (i + self._idx) % self._num_samples
    self:_fillCache(j)
    local cacheIdx = self:_global2cacheIndex(j)
    local img = self._cache_img[cacheIdx]
    -- Check the number of channels is correct
    n_channels = (n_channels or img:size()[1])
    assert(n_channels == img:size()[1], string.format(
    'Wrong number of channels in sample %q (got %d, expected %d)',
    k, img:size()[1], n_channels))
    batch_sizes[{i+1,1}] = img:size()[2]     -- Image height
    batch_sizes[{i+1,2}] = img:size()[3]     -- Image width
  end
  -- Copy data into single batch tensors
  local max_sizes = torch.max(batch_sizes, 1)
  if max_sizes[{1,2}] < self._min_width then
    max_sizes[{1,2}] = self._min_width
  end
  local batch_img = torch.Tensor(batch_size, n_channels, max_sizes[{1,1}],
          max_sizes[{1,2}]):zero()
  local batch_gt  = {}
  local batch_ids = {}
  local old_gpu = -1
  if self._cache_gpu >= 0 then
    old_gpu = cutorch.getDevice()
    cutorch.setDevice(self._cache_gpu)
    batch_img = batch_img:cuda()
  end
  for i=0,(batch_size-1) do
    local j = 1 + (i + self._idx) % self._num_samples
    self:_fillCache(j)
    local img = self._cache_img[self:_global2cacheIndex(j)]
    local gt  = self._cache_gt[self:_global2cacheIndex(j)]
    local dy = 0
    local dx = 0
    if self._centered_patch then
       dy = math.floor((max_sizes[{1,1}] - img:size()[2]) / 2)
       dx = math.floor((max_sizes[{1,2}] - img:size()[3]) / 2)
    end
    batch_img[i+1]:sub(1, n_channels,
       dy + 1, dy + img:size()[2],
       dx + 1, dx + img:size()[3]):copy(img)
    table.insert(batch_gt, gt)
    table.insert(batch_ids, self._samples[j])
  end
  if old_gpu >= 0 then
    cutorch.setDevice(old_gpu)
  end
  -- Increase index for next batch
  self._idx = (self._idx + batch_size) % self._num_samples
  collectgarbage()
  return batch_img, batch_gt, batch_sizes, batch_ids
end

function CachedBatcher:epochReset(img_list, cfg)
end

return CachedBatcher
