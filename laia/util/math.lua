-- Extension functions for Lua math.

-- Return true if a numeric value is NaN, otherwise returns false.
function math.isnan(x)
  return x ~= x
end

-- Return true if a numeric value is +/-inf, otherwise returns false.
function math.isinf(x)
  return x == math.huge or x == -math.huge
end

-- Return true if a numeric value is a valid finite number,
-- otherwise (NaN or +/-inf) returns false.
function math.isfinite(x)
  return x > -math.huge and x < math.huge
end

-- Override math.random to make sure that we use Torch random generator.
--
-- From the math package doc:
-- math.random() with no arguments generates a real number between 0 and 1.
-- math.random(upper) generates integer numbers between 1 and upper.
-- math.random(lower, upper) generates integer numbers between lower and upper.
local torch = require('torch')
math.random = function(a, b)
  if a ~= nil and b ~= nil then
    return torch.random(a, b)
  elseif a ~= nil then
    return torch.random(a)
  else
    return torch.uniform()
  end
end
