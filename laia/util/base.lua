laia = laia or {}
laia.log = laia.log or require 'laia.util.log'
laia.log.loglevel = 'warn'

-- Overload assert to use Laia's own logging system
assert = function(test, msg, ...)
  if not test then
    local function getfileline(filename, lineno)
      local n = 1
      for l in io.lines(filename) do
	if n == lineno then return l end
	n = n + 1
      end
    end
    -- Get the lua source code that caused the exception
    local info = debug.getinfo(2, 'Sl')
    local source = info.source
    if string.sub(source, 1, 1) == '@' then
      source = getfileline(string.sub(source, 2, #source),
			   info.currentline):gsub("^%s*(.-)%s*$", "%1")
    end
    msg = msg or ('Assertion %q failed'):format(source)
    laia.log.fatal{fmt = msg, arg = {...}, level = 3}
  end
end

-- Overload error to use Laia's own logging system
error = function(msg, ...)
  laia.log.fatal{fmt = msg, arg = {...}, level = 3}
end

-- Require with graceful warning, for optional modules
function wrequire(name)
  local ok, m = pcall(require, name)
  if not ok then
    laia.log.warn(('Optional lua module %q was not found!'):format(name))
  end
  return m or nil
end
