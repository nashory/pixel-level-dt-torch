-- this script will run server for display.
-- last modified : 2017.09.01, nashory


local opt = require 'opts'

if opt.display then
	os.execute(string.format('th -ldisplay.start %d %s', opt.display_port, opt.display_ip))
end


  
