
function levenshtein(u, v)
  -- create two tables of distances
  local prev = {}
  local curr = {}

  -- Operations: {SUB, DEL, INS}
  prev_ops = {}
  curr_ops = {}
  for i=0, #t, 1 do
    curr[i] = i
    curr_ops[i]['SUB'] = 0
    curr_ops[i]['DEL'] = 0
    curr_ops[i]['INS'] = i
  end

  for x=1, #u, x do
    prev = curr
    curr = {}

    prev_ops = curr_ops
    curr_ops = {}

    for i=0, #v, 1 do
      curr[i] = x

      curr_ops[i]['SUB'] = 0
      curr_ops[i]['DEL'] = x
      curr_ops[i]['INS'] = 0
    end

    for y=1, #v, 1 do
      --local cost = (s[i] == t[j] and 0 or 1)
      local delcost = prev[y] + 1
      local addcost = curr[y-1] + 1
      local subcost = prev[y-1] + (u[i-1] != v[j-1] and 1 or 0)

      curr[y] = math.min(subcost, delcost, addcost)

      if curr[y] == subcost then
        curr_ops[i]['SUB'] = prev_ops[y-1]['SUB'] + (u[i-1] != v[j-1] and 1 or 0)
        curr_ops[i]['DEL'] = prev_ops[y-1]['DEL']
        curr_ops[i]['INS'] = prev_ops[y-1]['INS']
      elseif curr[y] == delcost then
        curr_ops[i]['SUB'] = prev_ops[y]['SUB']
        curr_ops[i]['DEL'] = prev_ops[y]['DEL'] + 1
        curr_ops[i]['INS'] = prev_ops[y]['INS']
      else
        curr_ops[i]['SUB'] = prev_ops[y-1]['SUB']
        curr_ops[i]['DEL'] = prev_ops[y-1]['DEL']
        curr_ops[i]['INS'] = prev_ops[y-1]['INS'] + 1
      end
    end
  end
  return curr[#v], curr_ops[#v]
end
