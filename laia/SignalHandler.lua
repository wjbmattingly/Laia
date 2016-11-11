require ('laia.util.base')
local sig = require 'posix.signal'

local SignalHandler = torch.class('laia.SignalHandler')

SignalHandler.SIGINT = false
SignalHandler.SIGQUIT = false

sig.signal(sig.SIGINT,
	   function()
	     laia.log.debug('Catched SIGINT signal!')
	     SignalHandler.SIGINT = true
	   end, nil)
sig.signal(sig.SIGQUIT,
	   function()
	     laia.log.debug('Catched SIGQUIT signal!')
	     SignalHandler.SIGQUIT = true
	   end, nil)

function SignalHandler.ExitRequested()
  return SignalHandler.SIGINT or SignalHandler.SIGQUIT
end

return SignalHandler
