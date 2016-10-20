-- Extension functions for Lua string.

-- Return a list of the words in the input string, separated by whitespace
-- characters (space, tab, newline, etc). The optional argument pattern
-- can be used to separate the words using other delimiters.
--
-- Examples:
-- string.split('hello my friend')
-- { 'hello', 'my', 'friend' }
-- string.split('   hello     my    friend    ')
-- { 'hello', 'my', 'friend' }
-- string.split('hello,my,friend', '[^,]+')
-- { 'hello', 'my', 'friend' }
-- string.split('hello  ,  my  ,  friend', '[^,%s]+')
-- { 'hello', 'my', 'friend' }
string.split = function(str, pattern)
  assert(type(str) == 'string')
  pattern = pattern or "[^%s]+"
  if pattern:len() == 0 then pattern = "[^%s]+" end
  local parts = {__index = table.insert}
  setmetatable(parts, parts)
  str:gsub(pattern, parts)
  setmetatable(parts, nil)
  parts.__index = nil
  return parts
end
