laia = laia or {}
laia.mem = {}

local _cfg = {
  monitor_interval = 0,
  monitor_started  = false
}

--[[
  Function that returns the GPU memory used by the calling process.
  First, it tries to use the nvidia-smi command to get the exact number of
  memory used by the calling process.
  If this information is not available, it uses cutorch.getMemoryUsage().

  NOTE: This will only work on *nix systems, since it relies on /proc/self/stat,
  nvidia-smi and gawk.
--]]
local _PID = io.open('/proc/self/stat', 'r'):read('*number')
local _TO_MB = { KiB = 1024, MiB = 1, GiB = 1024, TiB = 1024 * 1024 }
local _CMD = ([[nvidia-smi | gawk '{
  if ($2 == "Processes:") { PF=1; }
  else if (PF && $3 == %d && match($(NF - 1), /^([0-9]+)(.iB)$/, A)) {
    if (A[2] == "KiB") S += A[1] / 1024;
    else if (A[2] == "MiB") S += A[1] * 1;
    else if (A[2] == "GiB") S += A[1] * 1024;
    else if (A[2] == "TiB") S += A[1] * 1024 * 1024;
  }
}END{ if (S > 0) print S; }']]):format(_PID)
function laia.mem.getCurrentGPUMemory()
  local nvidia_smi = io.popen(_CMD)
  local gpuMemory = (nvidia_smi ~= nil and nvidia_smi:read('*number')) or nil
  if not gpuMemory and cutorch ~= nil then
    local freeMemory, totalMemory = cutorch.getMemoryUsage()
    gpuMemory = (totalMemory - freeMemory) / (1024 * 1024)
  end
  return (gpuMemory or 0)
end

--[[
  Function that returns the resident CPU memory used by the calling process.
  We only monitor the resident size, since this is the actually amount that we
  care about (i.e. Lua uses an humongous amount of virtual memory).

  NOTE: This will only work on *nix systems, since it relies on getconf and
  /proc/self/statm.
]]--
local _PAGE_SIZE = io.popen('getconf PAGE_SIZE'):read('*number')
function laia.mem.getCurrentCPUMemory()
  local statmf = io.open('/proc/self/statm', 'r')
  statmf:read('*number')     -- Ignore VmSize
  local cpuMemory = statmf:read('*number') * _PAGE_SIZE / (1024 * 1024)
  statmf:close()
  return cpuMemory
end

local _maxCPUMemory, _maxGPUMemory = 0, 0
function laia.mem.getMaxCPUMemory()
  _maxCPUMemory = math.max(laia.mem.getCurrentCPUMemory(), _maxCPUMemory)
  return _maxCPUMemory
end

function laia.mem.getMaxGPUMemory()
  _maxGPUMemory = math.max(laia.mem.getCurrentGPUMemory(), _maxGPUMemory)
  return _maxGPUMemory
end

function laia.mem.registerOptions(parser, advanced)
  advanced = advanced or false
  if alarm then
    parser:option(
      '--memory_monitor_interval',
      'If n>0, monitorizes the memory usage every n seconds.',
      _cfg.monitor_interval, laia.toint)
      :argname('<n>')
      :bind(_cfg, 'monitor_interval')
      :advanced(advanced)
  end
end

wrequire('alarm')
local function _alarmMaxMemory()
  _maxCPUMemory = laia.mem.getMaxCPUMemory()
  _maxGPUMemory = laia.mem.getMaxGPUMemory()
  alarm(_cfg.monitor_interval)
end
function laia.mem.startMonitor()
  if alarm and not _cfg.monitor_started and _cfg.monitor_interval > 0 then
    alarm(_cfg.monitor_interval, _alarmMaxMemory)
  end
end
