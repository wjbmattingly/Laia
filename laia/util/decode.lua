-- Removes removes leading, trailing and repetitions of given symbol
function laia.symbol_trim(seq, symb)
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

function laia.levenshtein(u, v)
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

function laia.framewise_decode(batch_size, rnn_output)
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

-- This function returns a table conataining symbol ID sequence
-- corresponding to the force-alignment of the given ground-truth.
-- INPUT: Confidence Matrix Tensor containing the posterior
-- probabilities per character and frame, and Ground-Truth Vector
-- containing the char-ID sequence of the given sample.
-- OUTPUT: char-ID sequence alignment (it includes BLANK-CHAR)
function laia.force_alignment(teCM, tbGT)
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
  local traceBackPath = function (nS, nF, t)
    local aux = tbTBK[nS][nF]
    if aux[1]==0 and aux[2]==0 then
      table.insert(t,aux[3])
    else
      traceBackPath(aux[1],aux[2],t);
      table.insert(t,aux[3])
    end
  end

  if nSymbols > 1 and
  teTRL[{nSymbols-1,nframes}] > teTRL[{nSymbols,nframes}] then
    traceBackPath(nSymbols-1,nframes,t)
  else
    traceBackPath(nSymbols,nframes,t)
  end

  return t
end
