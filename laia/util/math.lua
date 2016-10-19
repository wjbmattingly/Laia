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
