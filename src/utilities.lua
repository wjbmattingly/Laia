--[[
   a = {1, 2, 3, 4}
   table.reduce(a, function(x) return 2* x; end)
   {2, 4, 6, 8}
--]]
table.map = function(t, fn)
   local t2 = {}
   for i, v in pairs(t) do
      t2[i] = fn(v)
   end
   return t2
end

--[[
   a = {1, 2, 3, 4}
   table.reduce(a, operator.add, 0)
   10
--]]
table.reduce = function(t, fn, v0)
   local res = v0
   for _, v in pairs(t) do
      res = fn(res, v)
   end
   return res
end

--[[
   Compute a histogram of the absolute values of the elements in a tensor,
   using a logarithmic scale.
   The function returs first a table containing the histogram (the number of
   elements in each bin), and secondly a table containing the thresholds for
   each bin.
--]]
torch.loghistc = function(x, nbins, bmin, bmax)
   x = torch.abs(x)
   nbins = nbins or 10
   bmin = bmin or torch.min(x)
   bmax = bmax or torch.max(x)
   if bmin == 0.0 then bmin = 1E-16 end
   assert(nbins > 1,
    string.format('nbins must be greater than 1 (actual: %d)', nbins))
   assert(bmax > bmin,
    string.format('bmax (%g) must be greater than bmin (%g)', bmax, bmin))
   local hist = { }
   local bins = { }
   local step = math.exp(math.log(bmax / bmin) / (nbins - 1))
   table.insert(bins, bmin)
   table.insert(hist, torch.le(x, bmin):sum())
   for i=1,(nbins-1) do
      local pb = bmin * math.pow(step, i - 1)
      local cb = pb * step
      table.insert(bins, cb)
      table.insert(hist, torch.cmul(torch.gt(x, pb), torch.le(x, cb)):sum())
   end
   -- Due to numerical errors, we may miss some elements in the last bin.
   local cumhist = table.reduce(hist, operator.add, 0)
   local missing = x:storage():size() - cumhist
   if missing > 0 then
      hist[#hist] = hist[#hist] + missing
   end
   return hist, bins
end

torch.sumarizeMagnitudes = function(x, mass, nbins, bmin, bmax, log_scale)
   mass = mass or 0.75
   log_scale = log_scale or true
   local hist, bins
   if log_scale then
      hist, bins = torch.loghistc(x, nbins, bmin, bmax)
   else
      x = torch.abs(x)
      bmin = bmin or torch.min(x)
      bmax = bmax or torch.max(x)
      hist = torch.totable(torch.histc(x, nbins, bmin, bmax))
      bins = torch.totable(torch.range(bmin, bmax, (bmax - bmin) / (nbins - 1)))
   end
   local n = x:storage():size()
   local aux = {}
   for i=1,#hist do table.insert(aux, {i, hist[i]})  end
   table.sort(aux,
        function(a, b)
     return a[2] > b[2] or (a[2] == b[2] and a[1] < b[1])
              end)
   local cum = 0
   local mini = #aux
   local maxi = 0
   for i=1,#aux do
      cum = cum + aux[i][2]
      if mini > aux[i][1] then mini = aux[i][1] end
      if maxi < aux[i][1] then maxi = aux[i][1] end
      if cum >= mass * n then break end
   end
   if mini < 2 then
      return torch.min(torch.abs(x)), bins[maxi], cum / n
   else
      return bins[mini - 1], bins[maxi], cum / n
   end
end

function levenshtein(u, v)
  -- create two tables of distances
  local prev = {}
  local curr = {}

  -- Operations: {SUB, DEL, INS}
  prev_ops = {}
  curr_ops = {}
  for i=0, #v, 1 do
   curr[i] = i

   curr_ops[i]={}

   curr_ops[i]['SUB'] = 0
   curr_ops[i]['DEL'] = 0
   curr_ops[i]['INS'] = i
  end

  for x=1, #u, 1 do
   prev = curr
   curr = {}

   prev_ops = curr_ops
   curr_ops = {}

   curr[0] = x
   curr_ops[0] = {}
   curr_ops[0]['SUB'] = 0
   curr_ops[0]['DEL'] = x
   curr_ops[0]['INS'] = 0

   for i=1, #v, 1 do
    curr_ops[i]={}
   end

   for y=1, #v, 1 do
    --local cost = (s[i] == t[j] and 0 or 1)
    local delcost = prev[y] + 1
    local addcost = curr[y-1] + 1
    local subcost = prev[y-1] + (u[x] ~= v[y] and 1 or 0)

    curr[y] = math.min(subcost, delcost, addcost)

    if curr[y] == subcost then
      curr_ops[y]['SUB'] = prev_ops[y-1]['SUB'] + (u[x] ~= v[y] and 1 or 0)
      curr_ops[y]['DEL'] = prev_ops[y-1]['DEL']
      curr_ops[y]['INS'] = prev_ops[y-1]['INS']
    elseif curr[y] == delcost then
      curr_ops[y]['SUB'] = prev_ops[y]['SUB']
      curr_ops[y]['DEL'] = prev_ops[y]['DEL'] + 1
      curr_ops[y]['INS'] = prev_ops[y]['INS']
    else
      curr_ops[y]['SUB'] = curr_ops[y-1]['SUB']
      curr_ops[y]['DEL'] = curr_ops[y-1]['DEL']
      curr_ops[y]['INS'] = curr_ops[y-1]['INS'] + 1
    end
   end
  end
  return curr[#v], curr_ops[#v]
end

function printHistogram(x, nbins, bmin, bmax)
  local hist, bins = torch.loghistc(x, nbins, bmin, bmax)
  local n = x:storage():size()
  io.write(string.format('(-inf, %g] -> %.2g%%\n',
        bins[1], 100 * hist[1] / n))
  for i=2,#hist do
   io.write(string.format('(%g, %g] -> %.2g%%\n',
          bins[i-1], bins[i], 100 * hist[i] / n))
  end
end

function framewise_decode(batch_size, rnn_output)
  local hyps = {}
  local seq_len = rnn_output:size(1) / batch_size
  local remove_blanks = remov_blanks or true
  -- Initialize hypotheses for each batch element
  for b=1,batch_size do
    table.insert(hyps, {})
  end

  local _, idx = torch.max(rnn_output,2)
  for b=0,batch_size-1 do
    for l=0, seq_len-1 do
      label = idx[l*batch_size+b+1][1]-1
      -- do not insert repeated labels
      if label ~= hyps[b+1][#hyps[b+1]] then
        table.insert(hyps[b+1], label)
      end
    end
  end
  -- remove 0's (BLANK) from decoding
  local hyps2 = {}
  for b=1,batch_size do
    table.insert(hyps2, {})
    for l=1,#hyps[b] do
      if hyps[b][l] ~= 0 then
        table.insert(hyps2[b], hyps[b][l])
      end
    end
  end
  return hyps2
end

-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end

-- split function for strings
string.split = function(str, pattern)
  pattern = pattern or "[^%s]+"
  if pattern:len() == 0 then pattern = "[^%s]+" end
  local parts = {__index = table.insert}
  setmetatable(parts, parts)
  str:gsub(pattern, parts)
  setmetatable(parts, nil)
  parts.__index = nil
  return parts
end

-- read symbols_table file. this file contains
-- two columns: "symbol     id"
function read_symbols_table(file)
  local lines = lines_from(file)
  -- check if exists or is not empty
  if lines == nil then return {} end
  --
  symbols_table = {}
  for i=1, #lines do
    local sline = lines[i]:split()
    symbol = sline[1]
    id = tonumber(sline[2])
    symbols_table[id] = symbol
  end
  return symbols_table
end

table.extend_with_last_element = function(t, n)
   n = n or (#t + 1)
   while #t < n do
      table.insert(t, t[#t])
   end
end

--[[
   Helper function used to sample a key from a table containing the likelihoods
   of each key element. This assumes that the scores in the likelihoods table
   are non-negative.
--]]
table.weighted_choice = function(l, z)
   z = z or table.reduce(l, operator.add, 0.0)
   local cut_likelihood = torch.uniform() * z
   local cum_likelihood = 0
   for k, w in pairs(l) do
      if cum_likelihood + w >= cut_likelihood then
	 return k
      end
      cum_likelihood = cum_likelihood + w
   end
   error('This point should not be reached. Make sure your that all your ' ..
	 'likelihoods are non-negative')
end

function save_gradInput_heatmap(m, desc)
   require 'image'
   desc = desc or ''
   assert(torch.isTypeOf(m, 'nn.Module'), 'Input must be a nn.Model')
   if torch.isTypeOf(m, 'nn.Container') then
      --for l=1,m:size() do
	 save_gradInput_heatmap(m:get(1), string.format('%s%d.', desc, 1))
      --end
   else
      local x = m.gradInput:clone()
      --print(string.format('%s %s', desc, torch.type(m)), x:size())
      for i=1,x:size()[1] do
	 local xi = x:sub(i, i):squeeze(1)
	 image.save(string.format('%s%d.jpg', desc, i), xi)
      end
   end
end
