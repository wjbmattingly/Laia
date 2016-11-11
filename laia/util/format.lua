laia = laia or {}

function laia.sec_to_dhms(sec)
  -- Split seconds into days (d), hours (h), minutes (m) seconds (s)
  local d = math.floor(sec / 86400)
  local h = math.floor((sec / 3600) % 24)
  local m = math.floor((sec / 60) % 60)
  local s = math.floor(sec % 60)
  local str = ''
  if d > 0 then str = ('%s%dd'):format(str, d) end
  if h > 0 then str = ('%s%dh'):format(str, h) end
  if m > 0 then str = ('%s%dm'):format(str, m) end
  if s > 0  or str == '' then str = ('%s%ds'):format(str, s) end
  return str
end
