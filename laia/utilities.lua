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
  a = {a = 1, b = 2, c = 3}
  b = {a = -1, d = -4}
  print(table.update_values(a, b))
  {
  a = -1,
  b = 2,
  c = 3
  }
--]]
table.update_values = function(dst, src)
  for k,_ in pairs(dst) do
    if src[k] ~= nil then dst[k] = src[k] end
  end
end

table.all = function(t, f)
  return table.reduce(t, function(p, x) return p and f(x) end, true)
end

table.any = function(t, f)
  return table.reduce(t, function(p, x) return p or f(x) end, true)
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
  log_scale = log_scale or false
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

-- Removes removes leading, trailing and repetitions of given symbol
function symbol_trim(seq, symb)
  local trimed = {}
  local N = #seq
  local n = 1
  while N > 0 and seq[N] == symb do
    N = N-1
  end
  while n <= N and seq[n] == symb do
    n = n+1
  end
  for m=n,N do
    if seq[m] ~= symb or seq[m] ~= seq[m-1] then
      table.insert(trimed,seq[m])
    end
  end
  return trimed
end

function levenshtein(u, v)
  -- create two tables of distances
  local prev = {}
  local curr = {}

  -- Operations: {sub, del, ins}
  prev_ops = {}
  curr_ops = {}
  for i=0, #v, 1 do
    curr[i] = i
    curr_ops[i] = {sub = 0, del = 0, ins = i}
  end

  for x=1, #u, 1 do
    prev = curr
    curr = {}

    prev_ops = curr_ops
    curr_ops = {}

    curr[0] = x
    curr_ops[0] = {sub = 0, del = x, ins = 0}

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
	curr_ops[y].sub = prev_ops[y-1].sub + (u[x] ~= v[y] and 1 or 0)
	curr_ops[y].del = prev_ops[y-1].del
	curr_ops[y].ins = prev_ops[y-1].ins
      elseif curr[y] == delcost then
	curr_ops[y].sub = prev_ops[y].sub
	curr_ops[y].del = prev_ops[y].del + 1
	curr_ops[y].ins = prev_ops[y].ins
      else
	curr_ops[y].sub = curr_ops[y-1].sub
	curr_ops[y].del = curr_ops[y-1].del
	curr_ops[y].ins = curr_ops[y-1].ins + 1
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
  local symbols_table = {}
  for i=1, #lines do
    local sline = lines[i]:split()
    local symbol = sline[1]
    local id = tonumber(sline[2])
    symbols_table[id] = symbol
  end
  return symbols_table
end

table.extend_with_last_element = function(t, n)
  n = n or (#t + 1)
  while #t < n do
    table.insert(t, t[#t])
  end
  return t
end

--[[
  Helper function used to sample a key from a table containing the likelihoods
  of each key element. This assumes that the scores in the likelihoods table
  are non-negative.
--]]
table.weighted_choice = function(l, z)
  z = z or table.reduce(l, operator.add, 0.0)
  -- If all weights are 0, sample randomly.
  if z <= 0.0 then return torch.random(#l) end
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

--[[
  -- Read a text file containing a matrix and load it into a Tensor
  function loadTensorFromFile(fname)
  local fd = io.open(fname, 'r')
  assert(fd ~= nil, string.format('Unable to read file: %q', fname))
  local tb = {}
  while true do
  local line = fd:read('*line')
  if line == nil then break end
  table.insert(tb, string.split(line))
  end
  fd:close()
  return torch.Tensor(tb)
  end
  -- Read a text file containing a vector IDs and load it into a Table
  function loadTableFromFile(fname)
  local fd = io.open(fname, 'r')
  assert(fd ~= nil, string.format('Unable to read file: %q', fname))
  local line = fd:read('*line')
  local tb = string.split(line)
  tb = torch.totable(torch.IntTensor(torch.IntStorage(tb)))
  fd:close()
  return tb
  end
--]]
-- This function returns a table conataining symbol ID sequence
-- corresponding to the force-alignment of the given ground-truth.
-- INPUT: Confidence Matrix Tensor containing the posterior
-- probabilities per character and frame, and Ground-Truth Vector
-- containing the char-ID sequence of the given sample.
-- OUTPUT: char-ID sequence alignment (it includes BLANK-CHAR)
function forceAlignment(teCM, tbGT)

  assert(teCM ~= nil and tbGT ~= nil, "Confidence matrix and/or ground-truth vector not defined")

  -- BLANK symbol ID
  local blkCharID = 1

  local nSymbols, nframes, nTotSymbols = #tbGT, teCM:size()[1], teCM:size()[2]
  local tbAuxGT = {}
  for i = 1, nSymbols do
    assert(tbGT[i]+1 <= nTotSymbols, string.format('One char-ID is greater than the total number of chars: %q',nTotSymbols))
    table.insert(tbAuxGT, blkCharID)
    -- tbAuxGT[1] is for blkCharID according to teCM[{{},1}], so we
    -- add 1 to every tbGT[i] (symb ID)
    table.insert(tbAuxGT, tbGT[i] + 1)
  end
  table.insert(tbAuxGT, blkCharID)
  nSymbols = #tbAuxGT

  -- teTRL: trellis tensor, tbTBK: trace back table
  local teTRL = torch.Tensor(nSymbols,nframes):fill(-1000)
  local tbTBK = {}
  tbTBK[1] = {}; tbTBK[2] = {};
  teTRL[{1,1}] = teCM[{1,tbAuxGT[1]}];
  tbTBK[1][1] = {0, 0, tbAuxGT[1]}
  teTRL[{2,1}] = teCM[{1,tbAuxGT[2]}];
  tbTBK[2][1] = {0, 0, tbAuxGT[2]}
  for s=3, nSymbols do
    tbTBK[s] = {}; tbTBK[s][1] = {0, 0, tbAuxGT[s]}
  end

  for f = 2, nframes do
    teTRL[{1,f}] = teTRL[{1,f-1}] + teCM[{f,tbAuxGT[1]}];
    tbTBK[1][f] = {1, f-1, tbAuxGT[1]}
    for s = 2, nSymbols do
      if teTRL[{s-1,f-1}] > teTRL[{s,f-1}] then
	teTRL[{s,f}] = teTRL[{s-1,f-1}]
	tbTBK[s][f] = {s-1, f-1, tbAuxGT[s]}
      else
	teTRL[{s,f}] = teTRL[{s,f-1}]
	tbTBK[s][f] = {s, f-1, tbAuxGT[s]}
      end
      if (s%2 == 0 and s>3) and (teTRL[{s-2,f-1}] > teTRL[{s,f}]) and (tbAuxGT[s-2] ~= tbAuxGT[s]) then
	teTRL[{s,f}] = teTRL[{s-2,f-1}]
	tbTBK[s][f] = {s-2, f-1, tbAuxGT[s]}
      end
      teTRL[{s,f}] = teTRL[{s,f}] + teCM[{f,tbAuxGT[s]}]
    end
  end

  local t = {}
  local traceBackPath
  traceBackPath = function (nS, nF, t)
    local aux = tbTBK[nS][nF]
    if aux[1]==0 and aux[2]==0 then
      table.insert(t,aux[3])
    else
      traceBackPath(aux[1],aux[2],t);
      table.insert(t,aux[3])
    end
  end

  if nSymbols > 1 and teTRL[{nSymbols-1,nframes}] > teTRL[{nSymbols,nframes}] then
    traceBackPath(nSymbols-1,nframes,t)
  else
    traceBackPath(nSymbols,nframes,t)
  end

  return t
end


-- from shell.lua, by Peter Odding
function shell_escape(...)
  local command = type(...) == 'table' and ... or { ... }
  for i, s in ipairs(command) do
    s = (tostring(s) or ''):gsub('"', '\\"')
    if s:find '[^A-Za-z0-9_."/-]' then
      s = '"' .. s .. '"'
    elseif s == '' then
      s = '""'
    end
    command[i] = s
  end
  return table.concat(command, ' ')
end

table.int2sym = function(t, int2sym)
  assert(torch.type(t) == 'table')
  local t2 = {}
  for i=1,#t do
    local int, sym = t[i], int2sym[t[i]]
    assert(sym ~= nil, string.format(
	     'Integer %d does not map to any symbol in the table', int))
    table.insert(t2, sym)
  end
  return t2
end

table.sym2int = function(t, sym2int)
  assert(torch.type(t) == 'table')
  local t2 = {}
  for i=1,#t do
    local sym, int = t[i], sym2int[t[i]]
    assert(int ~= nil, string.format(
	     'Symbol %q does not map to any integer in the table', sym))
    table.insert(t2, int)
  end
  return t2
end

table.from_iterator = function(...)
  local t = {}
  for e in ... do
    table.insert(t, e)
  end
  return t
end

function laia.isint(x)
  return (type(x) == 'number' and x == math.floor(x))
end

local __toboolean_table = {
  ['true']  = true,
  ['TRUE']  = true,
  ['T']     = true,
  ['1']     = true,
  ['false'] = false,
  ['FALSE'] = false,
  ['F']     = false,
  ['0']     = false
}
function laia.toboolean(x)
  return __toboolean_table[tostring(x)], ('value %q is not a boolean'):format(x)
end

-- Convert a variable value to an integer, if the value is not a valid integer
-- returns nil and an error message.
function laia.toint(x)
  local xn = tonumber(x)
  if laia.isint(xn) then return xn
  else return nil, ('value %q is not an integer'):format(x) end
end

-- Given a string that represents a list of NUMBERS separated by commas,
-- returns a table with the list items.
function laia.tolistnum(x)
  local sx = string.split(x, '[^,]+')
  local rx = {}
  for _, v in ipairs(sx) do
    local v2 = tonumber(v)
    if v2 == nil then
      return nil, ('value %q is not a number'):format(v)
    end
    table.insert(rx, v2)
  end
  return rx
end

-- Given a string that represents a list of INTEGERS separated by commas,
-- returns a table with the list items.
function laia.tolistint(x)
  local sx = string.split(x, '[^,]+')
  local rx = {}
  for _, v in ipairs(sx) do
    local v2 = laia.toint(v)
    if v2 == nil then
      return nil, ('value %q is not an integer'):format(v)
    end
    table.insert(rx, v2)
  end
  return rx
end

function laia.manualSeed(seed)
  torch.manualSeed(seed)
  if cutorch then cutorch.manualSeed(seed) end
end

function laia.getRNGState()
  local state = {}
  state.torch = torch.getRNGState()
  if cutorch then state.cutorch = cutorch.getRNGState() end
  return state
end

function laia.setRNGState(state)
  if state.torch then
    torch.setRNGState(state.torch)
  else
    laia.log.error('No torch RNG state found!')
  end
  if cutorch then
    if state.cutorch then
      cutorch.setRNGState(state.cutorch)
    else
      laia.log.error('No cutorch RNG state found!')
    end
  end
end

-- Return a flat view of a nn.Module parameters and gradients.
-- This assumes that m:getParameters() was called before to flatten
-- the parameters and gradients of the model.
function laia.getFlatParameters(m)
  assert(m ~= nil and torch.isTypeOf(m, 'nn.Module'),
	 ('Expected a nn.Module class (type = %q)'):format(torch.type(m)))
  local Tensor = torch.class(m:type()).new
  p, g = m:parameters()
  if not p or #p == 0 then return Tensor() end
  local sp, gp = p[1]:storage(), g[1]:storage()
  local n  = sp:size()
  return Tensor(sp, 1, n, 1), Tensor(gp, 1, n, 1)
end

-- Sample from a von Mises distribution, using rejection sampling
-- See: https://en.wikipedia.org/wiki/Von_Mises_distribution
--      https://en.wikipedia.org/wiki/Rejection_sampling
function laia.rand_von_Mises(m, k)
  local function lpdf(x, m, k)
    return k * math.cos(x - m) - math.log(2 * math.pi) - laia.log_bessel_I0(k)
  end
  local max, x = lpdf(m, m, k)
  repeat
    x = math.pi * (2.0 * torch.uniform() - 1.0)
  until math.log(torch.uniform()) + max < lpdf(x, m, k)
  return x
end

-- Compute log of the modified Bessel function of order 0, to avoid problems
-- with large values of x.
-- Approximation from: http://www.advanpix.com/2015/11/11/rational-approximations-for-the-modified-bessel-function-of-the-first-kind-i0-computations-double-precision
function laia.log_bessel_I0(x)
  -- Compute polynomial of degree n with coefficients stored in a
  -- y = \sum_{i=0}^n a_i x^i
  local function Poly(a, x)
    local y = a[1]
    for i=2,#a do y = y + a[i] * math.pow(x, i - 1) end
    return y
  end
  local P = {
    1.0000000000000000000000801e+00,
    2.4999999999999999999629693e-01,
    2.7777777777777777805664954e-02,
    1.7361111111111110294015271e-03,
    6.9444444444444568581891535e-05,
    1.9290123456788994104574754e-06,
    3.9367598891475388547279760e-08,
    6.1511873265092916275099070e-10,
    7.5940584360755226536109511e-12,
    7.5940582595094190098755663e-14,
    6.2760839879536225394314453e-16,
    4.3583591008893599099577755e-18,
    2.5791926805873898803749321e-20,
    1.3141332422663039834197910e-22,
    5.9203280572170548134753422e-25,
    2.0732014503197852176921968e-27,
    1.1497640034400735733456400e-29
  }
  local Q = {
    3.9894228040143265335649948e-01,
    4.9867785050353992900698488e-02,
    2.8050628884163787533196746e-02,
    2.9219501690198775910219311e-02,
    4.4718622769244715693031735e-02,
    9.4085204199017869159183831e-02,
   -1.0699095472110916094973951e-01,
    2.2725199603010833194037016e+01,
   -1.0026890180180668595066918e+03,
    3.1275740782277570164423916e+04,
   -5.9355022509673600842060002e+05,
    2.6092888649549172879282592e+06,
    2.3518420447411254516178388e+08,
   -8.9270060370015930749184222e+09,
    1.8592340458074104721496236e+11,
   -2.6632742974569782078420204e+12,
    2.7752144774934763122129261e+13,
   -2.1323049786724612220362154e+14,
    1.1989242681178569338129044e+15,
   -4.8049082153027457378879746e+15,
    1.3012646806421079076251950e+16,
   -2.1363029690365351606041265e+16,
    1.6069467093441596329340754e+16
  }
  local ax = math.abs(x)
  if ax < 7.75 then
    local z = ax * ax * 0.25
    return math.log(z * Poly(P, z) + 1)
  else
    return math.log(Poly(Q, 1.0 / ax)) + ax - 0.5 * math.log(ax)
  end
end
