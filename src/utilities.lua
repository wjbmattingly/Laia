
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
  local hyps = {};

  -- Initialize hypotheses for each batch element
  for i=1,batch_size do
    table.insert(hyps, {})
  end
  -- Put hypotheses with non repeated symbols
  local hyps = {};
  for t=1,#rnn_output do
    local _, idx = torch.max(rnn_output[t], 2)
    --print(idx[1])
    idx = torch.totable(idx - 1)
    --print(idx, _)
    for i=1,batch_size do
      if i <= #hyps then
        table.insert(hyps[i], idx[i][1])
      else
        hyps[i] = idx[i];
      end
    end
---    print(hyps)
  end
  local hyps2 = {}
  for i=1,batch_size do
    table.insert(hyps2, {})
    for t=1,#hyps[i] do
      is_ok = true
      -- if blank -> do not add 
      if hyps[i][t] == 0 then
        is_ok = false
      end
      -- if previous symbol is the same -> do not add
      if hyps[i][t] == hyps2[i][#hyps2[i]] then
        is_ok = false
      end      
      if is_ok then
        table.insert(hyps2[i], hyps[i][t])
      end
    end
  end
  return hyps2
end